from typing import List, Tuple
import numpy as np
import networkx as nx

from problem_base import Problem


def arc_cost(dist: float, load: float, alpha: float, beta: float) -> float:
    if dist <= 0:
        return 0.0
    return dist + (alpha * dist * load) ** beta


def build_dist_matrix_and_golds(problem: Problem) -> Tuple[np.ndarray, np.ndarray]:
    G = problem.graph
    n = G.number_of_nodes()
    dist_matrix = np.zeros((n, n), dtype=np.float64)
    golds = np.array([G.nodes[i]["gold"] for i in range(n)], dtype=np.float64)
    for i in range(n):
        lengths = nx.single_source_dijkstra_path_length(G, i, weight="dist")
        for j, d in lengths.items():
            dist_matrix[i, j] = d
    return dist_matrix, golds


def compute_prins_limit(
    dist_matrix: np.ndarray,
    golds: np.ndarray,
    alpha: float,
    beta: float,
    n: int,
) -> int:
    if n <= 1:
        return 1
    avg_dist_from_0 = np.mean(dist_matrix[0, 1:])
    total_nn = 0.0
    for i in range(1, n):
        row = np.delete(dist_matrix[i, :], i)
        row.sort()
        total_nn += row[1] if len(row) > 1 else 0.0
    avg_nn = total_nn / (n - 1) if (n - 1) > 0 else 0.0
    if avg_nn <= 0:
        return n
    avg_gold = float(np.percentile(golds[1:], 25.0))
    rhs = (
        2 * avg_dist_from_0
        - avg_nn
        + (avg_dist_from_0 * alpha * avg_gold) ** beta
    )
    denom = alpha * avg_nn
    if denom <= 0:
        return min(n, 50)
    w_limit = (rhs ** (1.0 / beta)) / denom
    k = max(5, int(w_limit / avg_gold)) if avg_gold > 0 else 50
    return min(k, n)


def optimal_split(
    tour: List[int],
    dist_matrix: np.ndarray,
    golds: np.ndarray,
    alpha: float,
    beta: float,
    limit: int,
) -> Tuple[float, List[int]]:
    N = len(tour)
    if N == 0:
        return 0.0, []
    V = np.full(N + 1, np.inf, dtype=np.float64)
    V[0] = 0.0
    predecessors = [-1] * (N + 1)
    for i in range(N):
        if V[i] == np.inf:
            continue
        load = 0.0
        curr_cost = 0.0
        prev = 0
        max_j = min(N, i + limit)
        for j in range(i, max_j):
            curr = tour[j]
            d_ij = dist_matrix[prev, curr]
            curr_cost += arc_cost(d_ij, load, alpha, beta)
            load += golds[curr]
            d_j0 = dist_matrix[curr, 0]
            return_cost = arc_cost(d_j0, load, alpha, beta)
            total_trip_cost = curr_cost + return_cost
            if V[i] + total_trip_cost < V[j + 1]:
                V[j + 1] = V[i] + total_trip_cost
                predecessors[j + 1] = i
            prev = curr
    return float(V[N]), predecessors


def predecessors_to_permutation(tour: List[int], predecessors: List[int]) -> List[int]:
    if not tour or predecessors is None or len(predecessors) != len(tour) + 1:
        return []
    out: List[int] = []
    curr = len(tour)
    segments: List[List[int]] = []
    while curr > 0:
        prev = predecessors[curr]
        seg = tour[prev:curr]
        segments.append(seg)
        curr = prev
    segments.reverse()
    for seg in segments:
        if out:
            out.append(0)
        out.extend(seg)
    return out


def prins_cost_and_permutation(
    problem: Problem,
    tour: List[int],
    *,
    limit: int | None = None,
    golds_override: np.ndarray | None = None,
) -> Tuple[float, List[int]]:
    dist_matrix, golds_from_problem = build_dist_matrix_and_golds(problem)
    golds = golds_override if golds_override is not None else golds_from_problem
    n = problem.graph.number_of_nodes()
    if limit is None:
        limit = compute_prins_limit(
            dist_matrix, golds, problem.alpha, problem.beta, n
        )
    cost_val, pred = optimal_split(
        tour, dist_matrix, golds, problem.alpha, problem.beta, limit
    )
    perm = predecessors_to_permutation(tour, pred)
    return cost_val, perm
