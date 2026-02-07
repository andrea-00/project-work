from typing import List, Union
import numpy as np
import networkx as nx

from problem_base import Problem
from .simulator import simulate_tour_partial

PartialItem = Union[int, tuple[int, float]]


def optimal_fraction_size(alpha: float, beta: float, distance_one_way: float) -> float:
    if beta <= 1.0 or distance_one_way <= 0:
        return float("inf")
    d = distance_one_way
    return (1.0 / alpha) * ((2.0 * (d ** (1.0 / beta))) / (beta - 1.0)) ** (1.0 / beta)


def build_partial_sequence(
    problem: Problem,
    permutation: List[int],
    *,
    high_gold_percentile: float = 50.0,
    use_optimal_fraction: bool = True,
    max_visits_per_city: int = 5,
) -> List[PartialItem]:
    G = problem.graph
    n = G.number_of_nodes()
    alpha, beta = problem.alpha, problem.beta
    dist_from_0 = {}
    for i in range(n):
        try:
            dist_from_0[i] = float(nx.shortest_path_length(G, 0, i, weight="dist"))
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            dist_from_0[i] = float("inf")
    out: List[PartialItem] = []
    for x in permutation:
        if x == 0:
            out.append(0)
            continue
        g = G.nodes[x]["gold"]
        if g <= 0:
            out.append((x, 1.0))
            continue
        if use_optimal_fraction and beta > 1.0:
            d = dist_from_0.get(x)
            if d is None or d <= 0:
                out.append((x, 1.0))
                continue
            L_star = optimal_fraction_size(alpha, beta, d)
            if L_star >= g or not np.isfinite(L_star):
                out.append((x, 1.0))
                continue
            remaining = g
            visits_done = 0
            while remaining > 1e-9 and visits_done < max_visits_per_city:
                take = min(remaining, L_star)
                out.append((x, take / g))
                remaining -= take
                visits_done += 1
                if remaining > 1e-9 and visits_done < max_visits_per_city:
                    out.append(0)
            if remaining > 1e-9:
                out.append(0)
                out.append((x, remaining / g))
        else:
            golds = [G.nodes[i]["gold"] for i in range(n)]
            threshold = float(np.percentile(golds, high_gold_percentile))
            if g > threshold:
                out.append((x, 0.5))
                out.append(0)
                out.append((x, 0.5))
            else:
                out.append((x, 1.0))
    return out


def cost_with_partial(
    problem: Problem,
    permutation: List[int],
    high_gold_percentile: float = 50.0,
) -> float:
    seq = build_partial_sequence(problem, permutation, high_gold_percentile=high_gold_percentile)
    return simulate_tour_partial(problem, seq, start_at_base=True, end_at_base=True)
