"""
Tour refinement methods for genetic algorithm solver.

Provides hill climbing, simulated annealing, tabu search,
and iterated local search for improving tours.
"""

import math
import random
from typing import List, Tuple

import numpy as np

from ...core.problem import Problem
from .split import build_distance_matrix, compute_split_limit, optimal_split


def _neighbor_swap(perm: List[int], rng: random.Random) -> List[int]:
    """Swap two random positions."""
    n = len(perm)
    if n < 2:
        return list(perm)
    i, j = rng.sample(range(n), 2)
    out = list(perm)
    out[i], out[j] = out[j], out[i]
    return out


def _neighbor_insert(perm: List[int], rng: random.Random) -> List[int]:
    """Move a random element to a new position."""
    n = len(perm)
    if n < 2:
        return list(perm)
    i = rng.randrange(n)
    j = rng.randrange(n)
    if i == j:
        return list(perm)
    out = list(perm)
    x = out.pop(i)
    out.insert(j, x)
    return out


def _neighbor_inversion(perm: List[int], rng: random.Random) -> List[int]:
    """Reverse a random segment."""
    n = len(perm)
    if n < 2:
        return list(perm)
    i, j = sorted(rng.sample(range(n), 2))
    out = list(perm)
    out[i : j + 1] = reversed(out[i : j + 1])
    return out


def propose_neighbor(perm: List[int], rng: random.Random) -> List[int]:
    """
    Propose a neighbor using random move selection.

    Move probabilities: 70% inversion, 20% insertion, 10% swap.
    """
    u = rng.random()
    if u < 0.70:
        return _neighbor_inversion(perm, rng)
    if u < 0.90:
        return _neighbor_insert(perm, rng)
    return _neighbor_swap(perm, rng)


def refine_tour(
    problem: Problem,
    perm: List[int],
    golds_override: np.ndarray | None = None,
    *,
    method: str = "hc",
    max_evals: int = 500,
    stall: int = 80,
    accept_equal: bool = True,
    rng: random.Random | None = None,
) -> Tuple[List[int], float]:
    """
    Refine a tour using local search.

    Args:
        problem: The problem instance
        perm: Initial tour permutation
        golds_override: Override gold values
        method: Search method ('hc', 'sa', 'tabu')
        max_evals: Maximum evaluations
        stall: Stall limit (stop if no improvement)
        accept_equal: Accept equal-cost solutions
        rng: Random number generator

    Returns:
        Tuple of (refined tour, cost)
    """
    if rng is None:
        rng = random.Random()

    if not perm:
        return [], 0.0

    dist_matrix, golds_from_p = build_distance_matrix(problem)
    golds = golds_override if golds_override is not None else golds_from_p
    n = problem.graph.number_of_nodes()
    limit = compute_split_limit(dist_matrix, golds, problem.alpha, problem.beta, n)

    def cost_fn(tour: List[int]) -> float:
        return optimal_split(tour, dist_matrix, golds, problem.alpha, problem.beta, limit)[0]

    best = list(perm)
    best_cost = cost_fn(best)
    cur = list(best)
    cur_cost = best_cost

    no_improve = 0
    evals = 0
    temperature = 1.0
    cooling = 0.995
    tabu_tenure = 8
    tabu_list: dict = {}

    while evals < max_evals:
        neigh = propose_neighbor(cur, rng)
        c = cost_fn(neigh)
        evals += 1
        delta = c - cur_cost

        # Acceptance based on method
        if method == "hc":
            accept = (delta < 0) or (accept_equal and delta == 0)
        elif method == "sa":
            accept = (delta < 0) or (rng.random() < math.exp(-delta / max(temperature, 1e-9)))
            temperature *= cooling
        else:  # tabu
            accept = delta < 0 or (accept_equal and delta == 0)

        if accept:
            cur = neigh
            cur_cost = c

        if cur_cost < best_cost:
            best = list(cur)
            best_cost = cur_cost
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= stall:
                break

    return best, best_cost


def perturb_tour(
    perm: List[int], strength: str = "medium", rng: random.Random | None = None
) -> List[int]:
    """
    Perturb a tour for diversification.

    Args:
        perm: Tour to perturb
        strength: Perturbation strength ('light', 'medium', 'heavy')
        rng: Random number generator

    Returns:
        Perturbed tour
    """
    if rng is None:
        rng = random.Random()

    out = list(perm)
    n = len(out)
    if n < 2:
        return out

    if strength == "heavy" and rng.random() < 0.5:
        for _ in range(rng.randint(1, 2)):
            i, j = sorted(rng.sample(range(n), 2))
            out[i : j + 1] = reversed(out[i : j + 1])
        return out

    k = {"light": (3, 4), "medium": (5, 6), "heavy": (7, 8)}.get(strength, (5, 6))
    num_swaps = rng.randint(k[0], k[1])

    for _ in range(num_swaps):
        i, j = rng.sample(range(n), 2)
        out[i], out[j] = out[j], out[i]

    return out


def iterated_local_search(
    problem: Problem,
    initial_perm: List[int],
    initial_cost: float,
    golds_override: np.ndarray | None = None,
    *,
    num_restarts: int = 3,
    refine_max_evals: int = 500,
    refine_stall: int = 80,
    perturb_strength: str = "medium",
    refine_after_perturb_evals: int = 150,
    rng: random.Random | None = None,
) -> Tuple[List[int], float]:
    """
    Iterated Local Search: alternate refinement and perturbation.

    Args:
        problem: The problem instance
        initial_perm: Starting tour
        initial_cost: Starting cost
        golds_override: Override gold values
        num_restarts: Number of perturbation-refinement cycles
        refine_max_evals: Max evals for initial refinement
        refine_stall: Stall limit for refinement
        perturb_strength: Perturbation strength
        refine_after_perturb_evals: Evals for post-perturbation refinement
        rng: Random number generator

    Returns:
        Tuple of (best tour, best cost)
    """
    if rng is None:
        rng = random.Random()

    # Initial refinement
    best, best_cost = refine_tour(
        problem,
        initial_perm,
        golds_override,
        method="hc",
        max_evals=refine_max_evals,
        stall=refine_stall,
        accept_equal=True,
        rng=rng,
    )

    for _ in range(num_restarts - 1):
        # Perturb
        perturbed = perturb_tour(best, strength=perturb_strength, rng=rng)

        # Refine
        refined, cost = refine_tour(
            problem,
            perturbed,
            golds_override,
            method="hc",
            max_evals=refine_after_perturb_evals,
            stall=refine_stall,
            accept_equal=True,
            rng=rng,
        )

        if cost < best_cost:
            best = refined
            best_cost = cost

    return best, best_cost