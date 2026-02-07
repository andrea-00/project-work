"""
Stitch dei tour intra-cluster in un'unica permutazione di nodi.

Dato l'ordine dei cluster e i tour intra per ogni cluster, trova i link
(exit/entry) a distanza minima tra cluster consecutivi, ruota i tour per
iniziare dall'entry, e concatena. Supporta ritorni a base (0) tra un cluster
e il successivo (Fase 5).
"""

from typing import List, Dict, Tuple, Optional
import numpy as np

from problem_base import Problem


def find_cheapest_link(
    nodes_a: List[int],
    nodes_b: List[int],
    metric: np.ndarray,
) -> Tuple[int, int]:
    """(exit_from_a, entry_to_b) con distanza minima."""
    if not nodes_a or not nodes_b:
        raise ValueError("empty node list")
    sub = metric[np.ix_(nodes_a, nodes_b)]
    r, c = np.unravel_index(np.argmin(sub), sub.shape)
    return nodes_a[r], nodes_b[c]


def stitch(
    cluster_order: List[int],
    intra_solutions: Dict[int, List[int]],
    distance_metric: np.ndarray,
    return_after_cluster: Optional[List[bool]] = None,
) -> List[int]:
    """
    Assembla il tour completo da ordine cluster e tour intra.

    Args:
        cluster_order: [0, c1, c2, ..., c_{k-1}] ordine di visita (0 = base).
        intra_solutions: {cluster_id: [nodi in ordine di visita]} (indici originali).
        distance_metric: matrice N×N distanze (per link tra cluster).
        return_after_cluster: se fornito, lista di k-1 bool. return_after_cluster[i]=True
            significa "dopo aver visitato cluster_order[i+1], torna a 0 (deposito) prima
            del prossimo cluster". Default: tutti False (nessun ritorno intermedio).

    Returns:
        Permutazione completa (può contenere 0 più volte se return_after_cluster lo richiede).
        simulate_tour con end_at_base=True aggiunge il ritorno finale a 0.
    """
    k = len(cluster_order)
    if k == 0:
        return []
    if k == 1:
        return list(intra_solutions.get(cluster_order[0], [0]))

    if return_after_cluster is None:
        return_after_cluster = [False] * (k - 1)
    if len(return_after_cluster) != k - 1:
        return_after_cluster = (return_after_cluster + [False] * (k - 1))[: k - 1]

    # Link (exit, entry) tra cluster consecutivi
    links: Dict[int, Tuple[int, int]] = {}
    for i in range(k):
        curr = cluster_order[i]
        next_c = cluster_order[(i + 1) % k]
        nodes_curr = intra_solutions[curr]
        nodes_next = intra_solutions[next_c]
        exit_n, entry_n = find_cheapest_link(nodes_curr, nodes_next, distance_metric)
        links[i] = (exit_n, entry_n)

    # Per ogni cluster: (entry, exit) per ruotare il tour
    link_points: Dict[int, Tuple[int, int]] = {}
    for i in range(k):
        cid = cluster_order[i]
        prev_i = (i - 1 + k) % k
        entry_n = links[prev_i][1]
        exit_n = links[i][0]
        link_points[cid] = (entry_n, exit_n)

    full_tour: List[int] = []
    for i, cid in enumerate(cluster_order):
        tour = intra_solutions[cid]
        entry_n, _ = link_points[cid]
        try:
            start_idx = tour.index(entry_n)
        except ValueError:
            start_idx = 0
        rotated = tour[start_idx:] + tour[:start_idx]
        full_tour.extend(rotated)
        # Ritorno a base dopo questo cluster (eccetto dopo l'ultimo, gestito da end_at_base)
        if i >= 1 and i - 1 < len(return_after_cluster) and return_after_cluster[i - 1]:
            full_tour.append(0)

    return full_tour
