from typing import List, Tuple
import numpy as np
import networkx as nx

from problem_base import Problem
from .partial import optimal_fraction_size
from .simulator import simulate_tour_partial

BulkSequence = List[int | Tuple[int, float]]


def remove_bulk(problem: Problem) -> Tuple[BulkSequence, np.ndarray]:
    G = problem.graph
    n = G.number_of_nodes()
    alpha, beta = problem.alpha, problem.beta
    remainder_golds = np.array([G.nodes[i]["gold"] for i in range(n)], dtype=np.float64)
    if beta <= 1.0:
        return [], remainder_golds
    dist_from_0 = {}
    for i in range(n):
        try:
            dist_from_0[i] = float(nx.shortest_path_length(G, 0, i, weight="dist"))
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            dist_from_0[i] = float("inf")
    bulk_sequence: BulkSequence = []
    for node in range(1, n):
        g = G.nodes[node]["gold"]
        if g <= 0:
            continue
        d = dist_from_0.get(node)
        if d is None or d <= 0:
            continue
        L_star = optimal_fraction_size(alpha, beta, d)
        if not np.isfinite(L_star) or L_star >= g:
            continue
        num_fractions = int(g // L_star)
        remainder_golds[node] = g % L_star
        frac = L_star / g
        for _ in range(num_fractions):
            bulk_sequence.append((node, frac))
            bulk_sequence.append(0)
    return bulk_sequence, remainder_golds


def bulk_cost(problem: Problem, bulk_sequence: BulkSequence) -> float:
    if not bulk_sequence:
        return 0.0
    return simulate_tour_partial(
        problem, bulk_sequence, start_at_base=True, end_at_base=True
    )
