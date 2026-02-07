import networkx as nx
from typing import List, Tuple

from problem_base import Problem


def path_cost_with_load(
    problem: Problem,
    source: int,
    target: int,
    load: float,
) -> Tuple[float, List[int]]:
    G = problem.graph
    if source == target:
        return 0.0, [source]
    def weight(u, v, d, load=load):
        return problem.cost([u, v], load)
    try:
        path = nx.dijkstra_path(G, source, target, weight=weight)
    except nx.NetworkXNoPath:
        return float("inf"), []
    cost = 0.0
    for a, b in zip(path, path[1:]):
        cost += problem.cost([a, b], load)
    return cost, path


def simulate_tour(
    problem: Problem,
    permutation: List[int],
    *,
    start_at_base: bool = True,
    end_at_base: bool = True,
) -> float:
    G = problem.graph
    current = 0
    load = 0.0
    total_cost = 0.0
    if not permutation:
        if end_at_base and current != 0:
            c, _ = path_cost_with_load(problem, current, 0, load)
            total_cost += c
        return total_cost
    if start_at_base:
        next_node = permutation[0]
        if next_node != 0:
            c, _ = path_cost_with_load(problem, 0, next_node, 0.0)
            total_cost += c
            load = G.nodes[next_node]["gold"]
            current = next_node
        start_i = 1
    else:
        current = permutation[0]
        load = G.nodes[current]["gold"] if current != 0 else 0.0
        start_i = 1
    for i in range(start_i, len(permutation)):
        next_node = permutation[i]
        if next_node == 0:
            c, _ = path_cost_with_load(problem, current, 0, load)
            total_cost += c
            load = 0.0
            current = 0
        else:
            c, _ = path_cost_with_load(problem, current, next_node, load)
            total_cost += c
            load += G.nodes[next_node]["gold"]
            current = next_node
    if end_at_base and current != 0:
        c, _ = path_cost_with_load(problem, current, 0, load)
        total_cost += c
    return total_cost


_Visit = tuple[int, float]


def simulate_tour_partial(
    problem: Problem,
    sequence: List[int | _Visit],
    *,
    start_at_base: bool = True,
    end_at_base: bool = True,
) -> float:
    G = problem.graph
    current = 0
    load = 0.0
    total_cost = 0.0
    if not sequence:
        if end_at_base and current != 0:
            c, _ = path_cost_with_load(problem, current, 0, load)
            total_cost += c
        return total_cost
    i = 0
    if start_at_base and i < len(sequence):
        elem = sequence[i]
        if elem == 0:
            current, load = 0, 0.0
            i += 1
        elif isinstance(elem, tuple):
            node, frac = elem
            c, _ = path_cost_with_load(problem, 0, node, 0.0)
            total_cost += c
            load = frac * G.nodes[node]["gold"]
            current = node
            i += 1
    while i < len(sequence):
        elem = sequence[i]
        if elem == 0:
            c, _ = path_cost_with_load(problem, current, 0, load)
            total_cost += c
            load = 0.0
            current = 0
            i += 1
        elif isinstance(elem, tuple):
            node, frac = elem
            c, _ = path_cost_with_load(problem, current, node, load)
            total_cost += c
            load += frac * G.nodes[node]["gold"]
            current = node
            i += 1
        else:
            i += 1
    if end_at_base and current != 0:
        c, _ = path_cost_with_load(problem, current, 0, load)
        total_cost += c
    return total_cost
