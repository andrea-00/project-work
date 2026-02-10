"""
Adaptive solver for the Gold Collection Problem.

Automatically selects the best solving regime based on problem parameters
and orchestrates the solution pipeline.
"""

import random
from enum import Enum
from typing import Literal, Optional

import numpy as np

from ..core.problem import Problem
from ..core.solution import Trip, Solution
from ..core.evaluator import Evaluator
from ..distance.oracle import profile, build_oracle, LandmarkOracle
from .heuristics.constructive import (
    split_tour,
    construct_tour_regime_t,
    construct_solution_regime_l,
    construct_solution_regime_s,
)
from .heuristics.local_search import lns_improve, LNSParams


class Regime(Enum):
    """
    Solving regime based on cost function characteristics.

    T: Tour-based for sub-linear costs (beta < 1)
    L: Linear/star-based for linear costs (beta = 1)
    S: Soft-capacity for super-linear costs (beta > 1)
    """

    T = "T"  # Tour-based (sub-linear)
    L = "L"  # Linear (star-based)
    S = "S"  # Soft-capacity (super-linear)


def _choose_regime(problem: Problem, oracle: LandmarkOracle) -> Regime:
    """
    Select the appropriate solving regime.

    Args:
        problem: The problem instance
        oracle: Distance oracle

    Returns:
        Selected regime (T, L, or S)
    """
    beta = problem.beta

    if beta > 1.0:
        return Regime.S
    if beta < 1.0:
        return Regime.T
    return Regime.L


def _get_split_window(n: int, tour_len: int) -> int:
    """Determine DP window size based on problem size."""
    if n <= 500:
        return tour_len
    if n <= 2000:
        return min(800, tour_len)
    return min(500, tour_len)


def solve_return_solution(
    problem: Problem,
    *,
    time_budget_s: float = 1200.0,
    rng_seed: Optional[int] = None,
) -> Solution:
    """
    Solve the Gold Collection Problem and return the Solution object.

    Same as solve() but returns the solution (trips) instead of the cost.
    Used when the caller needs the path and item choices (e.g. for submission format).

    Args:
        problem: The problem instance
        time_budget_s: Total time budget in seconds
        rng_seed: Random seed for reproducibility

    Returns:
        Solution with trips and unserved; may have empty trips if unsolvable.
    """
    if rng_seed is not None:
        random.seed(rng_seed)
    rng = random.Random(rng_seed)

    plan = profile(problem)
    oracle = build_oracle(problem, plan=plan)
    evaluator = Evaluator(problem, oracle)

    G = problem.graph
    n = G.number_of_nodes()
    regime = _choose_regime(problem, oracle)

    if regime == Regime.T:
        sol = _solve_regime_t(problem, oracle, evaluator, time_budget_s, n, rng)
    elif regime == Regime.L:
        sol = _solve_regime_l(problem, evaluator, time_budget_s, rng_seed, rng)
    else:
        sol = _solve_regime_s(problem, evaluator, time_budget_s, rng)

    return sol


def solve(
    problem: Problem,
    *,
    time_budget_s: float = 1200.0,
    use_exact_final: bool = True,
    rng_seed: Optional[int] = None,
) -> float:
    """
    Solve the Gold Collection Problem.

    Automatically selects the best regime and applies construction
    and improvement heuristics.

    Args:
        problem: The problem instance
        time_budget_s: Total time budget in seconds
        use_exact_final: Use exact cost for final evaluation
        rng_seed: Random seed for reproducibility

    Returns:
        Total solution cost
    """
    sol = solve_return_solution(
        problem,
        time_budget_s=time_budget_s,
        rng_seed=rng_seed,
    )

    if not sol.trips and not sol.unserved:
        return problem.baseline()

    plan = profile(problem)
    oracle = build_oracle(problem, plan=plan)
    evaluator = Evaluator(problem, oracle)
    if use_exact_final:
        return evaluator.exact_solution_cost(sol)
    return evaluator.solution_cost(sol)


def _solve_regime_t(
    problem: Problem,
    oracle: LandmarkOracle,
    evaluator: Evaluator,
    time_budget_s: float,
    n: int,
    rng: random.Random,
) -> Solution:
    """Solve using Regime T (tour-based for sub-linear costs)."""
    tour = construct_tour_regime_t(problem, oracle, rng=rng)

    if not tour:
        return Solution(trips=[], unserved={})

    window_W = _get_split_window(n, len(tour))
    sol = split_tour(problem, tour, evaluator, window_W=window_W)

    # Apply LNS with moderate parameters
    sol = lns_improve(
        problem,
        sol,
        evaluator,
        LNSParams(destroy_fraction=0.10, regret_k=2, max_iter=50),
        time_budget_s=min(time_budget_s * 0.5, 600),
        rng=rng,
    )

    return sol


def _solve_regime_l(
    problem: Problem,
    evaluator: Evaluator,
    time_budget_s: float,
    rng_seed: Optional[int],
    rng: random.Random,
) -> Solution:
    """Solve using Regime L (star-based for linear costs)."""
    lns_time = min(time_budget_s * 0.50, 500)
    n_restarts = 3

    best_sol = None
    best_cost = float("inf")

    for start in range(n_restarts):
        r = random.Random((rng_seed if rng_seed is not None else 0) + 1 + start)

        cand = construct_solution_regime_l(problem, evaluator, rng=r)

        # Aggressive LNS with simulated annealing and merge/split
        cand = lns_improve(
            problem,
            cand,
            evaluator,
            LNSParams(
                destroy_fraction=0.22,
                regret_k=3,
                max_iter=80,
                use_sa=True,
                sa_initial_temp=0.08,
                sa_cooling=0.993,
                enable_merge_split=True,
                merge_split_tries=35,
            ),
            time_budget_s=lns_time / n_restarts,
            rng=r,
        )

        c = evaluator.solution_cost(cand)
        if c < best_cost:
            best_cost = c
            best_sol = cand

    return best_sol if best_sol is not None else construct_solution_regime_l(
        problem, evaluator, rng=rng
    )


def _solve_regime_s(
    problem: Problem,
    evaluator: Evaluator,
    time_budget_s: float,
    rng: random.Random,
) -> Solution:
    """Solve using Regime S (soft-capacity for super-linear costs)."""
    sol = construct_solution_regime_s(problem, evaluator, q_percentile=30.0, rng=rng)

    # Handle any unserved demand
    if sol.unserved:
        for i, amt in list(sol.unserved.items()):
            if amt > 1e-9:
                sol.trips.append(Trip(stops=[0, i, 0], pickups=[0.0, amt, 0.0]))
        sol.unserved = {}

    # Light LNS
    sol = lns_improve(
        problem,
        sol,
        evaluator,
        LNSParams(destroy_fraction=0.10, regret_k=2, max_iter=10),
        time_budget_s=min(time_budget_s * 0.5, 600),
        rng=rng,
    )

    return sol


def get_solution_cost(problem: Problem, time_budget_s: float = 120.0) -> float:
    """
    Convenience function to get solution cost for a problem.

    Args:
        problem: The problem instance
        time_budget_s: Time budget in seconds

    Returns:
        Optimized solution cost
    """
    return solve(problem, time_budget_s=time_budget_s)