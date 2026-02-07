"""
Fase 5: ottimizzazione dei ritorni a base (0).

Vettore di bit: return_after_cluster[i] = True significa "dopo il cluster i+1, torna a 0
prima di proseguire". Hill Climbing e Simulated Annealing (Cap. 3, 4) sul vettore di bit.
"""

import random
import math
from typing import List, Callable

from problem_base import Problem
from .linker import stitch
from .simulator import simulate_tour


def cost_with_returns(
    problem: Problem,
    cluster_order: List[int],
    intra_solutions: dict,
    distance_metric,
    return_bits: List[bool],
) -> float:
    """Costo del tour con i ritorni a 0 dati da return_bits (lunghezza k-1)."""
    full = stitch(
        cluster_order,
        intra_solutions,
        distance_metric,
        return_after_cluster=return_bits,
    )
    return simulate_tour(problem, full, start_at_base=True, end_at_base=True)


def hill_climbing_returns(
    problem: Problem,
    cluster_order: List[int],
    intra_solutions: dict,
    distance_metric,
    max_iters: int = 100,
    start_all_true: bool = True,
    rng_seed: int | None = None,
) -> tuple[List[bool], float]:
    """
    Hill Climbing sul vettore di bit. start_all_true=True parte da "ritorna dopo ogni
    cluster" (carico basso per viaggio), spesso migliore della baseline.

    Returns:
        (best_return_bits, best_cost).
    """
    k = len(cluster_order)
    n_bits = k - 1
    if n_bits <= 0:
        return [], cost_with_returns(
            problem, cluster_order, intra_solutions, distance_metric, []
        )

    if rng_seed is not None:
        random.seed(rng_seed)

    bits = [start_all_true] * n_bits
    cost = cost_with_returns(problem, cluster_order, intra_solutions, distance_metric, bits)
    for _ in range(max_iters):
        improved = False
        for i in range(n_bits):
            bits[i] = not bits[i]
            c = cost_with_returns(
                problem, cluster_order, intra_solutions, distance_metric, bits
            )
            if c < cost:
                cost = c
                improved = True
            else:
                bits[i] = not bits[i]
        if not improved:
            break
    return list(bits), cost


def simulated_annealing_returns(
    problem: Problem,
    cluster_order: List[int],
    intra_solutions: dict,
    distance_metric,
    max_iters: int = 200,
    t_start: float = 1000.0,
    t_end: float = 0.1,
    start_all_true: bool = True,
    rng_seed: int | None = None,
) -> tuple[List[bool], float]:
    """
    Simulated Annealing sul vettore di bit: vicino = flip un bit casuale.
    Accetta peggioramenti con prob exp(-delta/T). T decresce linearmente.

    Returns:
        (best_return_bits, best_cost).
    """
    k = len(cluster_order)
    n_bits = k - 1
    if n_bits <= 0:
        return [], cost_with_returns(
            problem, cluster_order, intra_solutions, distance_metric, []
        )

    if rng_seed is not None:
        random.seed(rng_seed)

    bits = [start_all_true] * n_bits
    cost = cost_with_returns(problem, cluster_order, intra_solutions, distance_metric, bits)
    best_bits = list(bits)
    best_cost = cost

    for it in range(max_iters):
        t = t_start + (t_end - t_start) * (it / max(max_iters - 1, 1))
        i = random.randrange(n_bits)
        bits[i] = not bits[i]
        new_cost = cost_with_returns(
            problem, cluster_order, intra_solutions, distance_metric, bits
        )
        delta = new_cost - cost
        if delta <= 0 or random.random() < math.exp(-delta / max(t, 1e-9)):
            cost = new_cost
            if cost < best_cost:
                best_cost = cost
                best_bits = list(bits)
        else:
            bits[i] = not bits[i]

    return best_bits, best_cost
