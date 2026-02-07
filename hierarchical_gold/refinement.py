import math
import random
from typing import Callable, List, Tuple

import numpy as np

from problem_base import Problem
from .prins import build_dist_matrix_and_golds, compute_prins_limit, optimal_split


def _neighbor_swap(perm: List[int], rng: random.Random) -> List[int]:
    n = len(perm)
    if n < 2:
        return list(perm)
    i, j = rng.sample(range(n), 2)
    out = list(perm)
    out[i], out[j] = out[j], out[i]
    return out


def _neighbor_insert(perm: List[int], rng: random.Random) -> List[int]:
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
    n = len(perm)
    if n < 2:
        return list(perm)
    i, j = sorted(rng.sample(range(n), 2))
    out = list(perm)
    out[i : j + 1] = reversed(out[i : j + 1])
    return out


def propose_neighbor(perm: List[int], rng: random.Random) -> List[int]:
    u = rng.random()
    if u < 0.70:
        return _neighbor_inversion(perm, rng)
    if u < 0.90:
        return _neighbor_insert(perm, rng)
    return _neighbor_swap(perm, rng)


def _propose_neighbor_with_move(
    perm: List[int], rng: random.Random
) -> Tuple[List[int], Tuple[str, int, int]]:
    u = rng.random()
    n = len(perm)
    if u < 0.70:
        i, j = sorted(rng.sample(range(n), 2))
        out = list(perm)
        out[i : j + 1] = reversed(out[i : j + 1])
        return out, ("inv", i, j)
    if u < 0.90:
        i, j = rng.randrange(n), rng.randrange(n)
        if i == j:
            j = (j + 1) % n
        out = list(perm)
        x = out.pop(i)
        out.insert(j, x)
        return out, ("ins", i, j)
    i, j = rng.sample(range(n), 2)
    out = list(perm)
    out[i], out[j] = out[j], out[i]
    return out, ("swp", min(i, j), max(i, j))


def refine_tour_prins(
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
    if rng is None:
        rng = random.Random()
    if not perm:
        return [], 0.0
    dist_matrix, golds_from_p = build_dist_matrix_and_golds(problem)
    golds = golds_override if golds_override is not None else golds_from_p
    n = problem.graph.number_of_nodes()
    limit = compute_prins_limit(
        dist_matrix, golds, problem.alpha, problem.beta, n
    )

    def cost_fn(tour: List[int]) -> float:
        return optimal_split(
            tour, dist_matrix, golds,
            problem.alpha, problem.beta, limit
        )[0]

    best = list(perm)
    best_cost = cost_fn(best)
    cur = list(best)
    cur_cost = best_cost
    no_improve = 0
    evals = 0
    T = 1.0
    cooling = 0.995
    tabu_tenure = 8
    tabu_list: dict[Tuple[str, int, int], int] = {}

    while evals < max_evals:
        if method == "tabu":
            neigh, move_key = _propose_neighbor_with_move(cur, rng)
        else:
            neigh = propose_neighbor(cur, rng)
            move_key = ()
        c = cost_fn(neigh)
        evals += 1
        delta = c - cur_cost
        if method == "hc":
            accept = (delta < 0) or (accept_equal and delta == 0)
        elif method == "sa":
            accept = (delta < 0) or (
                rng.random() < math.exp(-delta / max(T, 1e-9))
            )
            T *= cooling
        else:
            is_tabu = move_key and tabu_list.get(move_key, 0) > evals
            aspiration = c < best_cost
            accept = (delta < 0 or (accept_equal and delta == 0)) and (not is_tabu or aspiration)
            if accept and move_key:
                tabu_list[move_key] = evals + tabu_tenure
            for k in list(tabu_list):
                if tabu_list[k] <= evals:
                    del tabu_list[k]
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


def apply_lns(
    perm: List[int],
    cost_fn: Callable[[List[int]], float],
    remove_percent: float = 0.25,
    rng: random.Random | None = None,
) -> List[int]:
    if rng is None:
        rng = random.Random()
    n = len(perm)
    if n <= 2:
        return list(perm)
    k = max(1, min(n - 1, int(n * remove_percent)))
    idx_remove = set(rng.sample(range(n), k))
    removed = [perm[i] for i in sorted(idx_remove)]
    partial = [perm[i] for i in range(n) if i not in idx_remove]
    rng.shuffle(removed)
    for city in removed:
        best_cost = float("inf")
        best_pos = 0
        for pos in range(len(partial) + 1):
            candidate = partial[:pos] + [city] + partial[pos:]
            c = cost_fn(candidate)
            if c < best_cost:
                best_cost = c
                best_pos = pos
        partial = partial[:best_pos] + [city] + partial[best_pos:]
    return partial


def perturb_tour(perm: List[int], strength: str = "medium", rng: random.Random | None = None) -> List[int]:
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
    if rng is None:
        rng = random.Random()
    best = list(initial_perm)
    best_cost = initial_cost
    best, best_cost = refine_tour_prins(
        problem, best, golds_override,
        method="hc", max_evals=refine_max_evals, stall=refine_stall,
        accept_equal=True, rng=rng,
    )
    for _ in range(num_restarts - 1):
        perturbed = perturb_tour(best, strength=perturb_strength, rng=rng)
        refined, cost = refine_tour_prins(
            problem, perturbed, golds_override,
            method="hc", max_evals=refine_after_perturb_evals, stall=refine_stall,
            accept_equal=True, rng=rng,
        )
        if cost < best_cost:
            best = refined
            best_cost = cost
    return best, best_cost
