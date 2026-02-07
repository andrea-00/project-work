"""
Factory: costruzione del meta-problem (k×k) e degli intra-problem per cluster.

Il meta-problem ha fitness = -simulate_tour(stitch(cluster_order, intra_tours)).
Gli intra-problem hanno entry_load e entry_node per la fitness del sotto-tour.
"""

from typing import Dict, List, Tuple, Callable
import numpy as np

from problem_base import Problem
from .gold_problem_adapter import GoldCollectionAdapter
from .simulator import simulate_tour
from .linker import stitch, find_cheapest_link


def build_meta_matrix(
    distance_metric: np.ndarray,
    cluster_map: Dict[int, List[int]],
    k: int,
) -> np.ndarray:
    """
    Matrice k×k: meta[i][j] = minima distanza tra un nodo del cluster i e uno del cluster j.
    """
    meta = np.zeros((k, k))
    for i in range(k):
        for j in range(k):
            if i == j:
                meta[i, j] = 0.0
                continue
            nodes_i = cluster_map[i]
            nodes_j = cluster_map[j]
            min_d = float("inf")
            for ni in nodes_i:
                for nj in nodes_j:
                    d = distance_metric[ni, nj]
                    if d < min_d:
                        min_d = d
            meta[i, j] = min_d
    return meta


def nearest_neighbor_intra(
    problem: Problem,
    nodes: List[int],
    start_node: int,
    distance_metric: np.ndarray,
) -> List[int]:
    """
    Ordine di visita dei nodi (sottoinsieme) con Nearest Neighbor da start_node.
    Restituisce lista di indici originali.
    """
    if not nodes:
        return []
    if len(nodes) == 1:
        return list(nodes)
    order = [start_node]
    unvisited = set(nodes) - {start_node}
    current = start_node
    while unvisited:
        best = None
        best_d = float("inf")
        for u in unvisited:
            d = distance_metric[current, u]
            if d < best_d:
                best_d = d
                best = u
        if best is None:
            break
        order.append(best)
        current = best
        unvisited.discard(best)
    return order


def heuristic_intra_tours(
    adapter: GoldCollectionAdapter,
    cluster_map: Dict[int, List[int]],
    distance_metric: np.ndarray,
    problem: Problem,
) -> Dict[int, List[int]]:
    """
    Tour intra euristici (NN) per ogni cluster: partenza dal nodo più vicino a 0.
    """
    intra: Dict[int, List[int]] = {}
    for cid, node_list in cluster_map.items():
        if cid == 0:
            intra[0] = [0]
            continue
        if not node_list:
            continue
        # Nodo del cluster più vicino a 0
        start = min(node_list, key=lambda n: distance_metric[0, n])
        intra[cid] = nearest_neighbor_intra(problem, node_list, start, distance_metric)
    return intra


def meta_fitness_builder(
    problem: Problem,
    adapter: GoldCollectionAdapter,
    cluster_map: Dict[int, List[int]],
    k: int,
) -> Callable[[List[int]], float]:
    """
    Restituisce la funzione fitness per il meta-GA.

    genotype = permutazione di [1, 2, ..., k-1] (ordine dei cluster non-base).
    cluster_order = [0] + genotype (partenza da base, poi cluster in ordine).
    Fitness = -simulate_tour(stitch(cluster_order, heuristic_intra)).
    """
    metric = adapter.get_cost_metric()

    def fitness(genotype: List[int]) -> float:
        cluster_order = [0] + list(genotype)
        intra = heuristic_intra_tours(adapter, cluster_map, metric, problem)
        full = stitch(cluster_order, intra, metric)
        cost = simulate_tour(problem, full, start_at_base=True, end_at_base=True)
        return -cost

    return fitness


def compute_entry_loads_and_nodes(
    problem: Problem,
    cluster_order: List[int],
    intra_solutions: Dict[int, List[int]],
    distance_metric: np.ndarray,
    cluster_map: Dict[int, List[int]],
) -> Tuple[Dict[int, float], Dict[int, int], Dict[int, int]]:
    """
    Dato l'ordine cluster e i tour intra, calcola per ogni cluster:
    - entry_load: carico quando si entra nel cluster
    - entry_node: nodo di ingresso
    - exit_node: nodo di uscita (per il linker, già noto)

    Restituisce (entry_loads, entry_nodes, exit_nodes).
    """
    G = problem.graph
    entry_loads: Dict[int, float] = {}
    entry_nodes: Dict[int, int] = {}
    exit_nodes: Dict[int, int] = {}

    # Link tra cluster consecutivi
    k = len(cluster_order)
    links = []
    for i in range(k):
        curr = cluster_order[i]
        next_c = cluster_order[(i + 1) % k]
        nodes_curr = intra_solutions[curr]
        nodes_next = intra_solutions[next_c]
        ex, en = find_cheapest_link(nodes_curr, nodes_next, distance_metric)
        links.append((ex, en))

    # Carico accumulato: partiamo da 0 con load 0
    load = 0.0
    for i in range(k):
        cid = cluster_order[i]
        entry_n = links[(i - 1 + k) % k][1]
        exit_n = links[i][0]
        entry_nodes[cid] = entry_n
        exit_nodes[cid] = exit_n
        entry_loads[cid] = load
        # Dopo aver visitato il cluster, load += oro di tutti i nodi del cluster
        for node in intra_solutions[cid]:
            load += G.nodes[node]["gold"]

    return entry_loads, entry_nodes, exit_nodes
