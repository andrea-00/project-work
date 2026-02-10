"""
Large Neighbourhood Search (LNS) for solution improvement.

Implements destroy-repair with regret-k insertion and local search operators.
"""

import math
import random
import time
from dataclasses import dataclass
from typing import List, Optional, Tuple

from ...core.problem import Problem
from ...core.solution import Trip, Solution
from ...core.evaluator import Evaluator


@dataclass
class LNSParams:
    """
    Parameters for Large Neighbourhood Search.

    Attributes:
        destroy_fraction: Fraction of stops to remove (0.0 to 1.0)
        regret_k: Number of insertion options to consider
        max_iter: Maximum iterations (None = unlimited)
        use_sa: Enable simulated annealing acceptance
        sa_initial_temp: Initial temperature for SA
        sa_cooling: Cooling rate for SA
        enable_merge_split: Enable merge/split operator
        merge_split_tries: Number of merge/split attempts per iteration
    """

    destroy_fraction: float = 0.10
    regret_k: int = 2
    max_iter: Optional[int] = None
    use_sa: bool = False
    sa_initial_temp: float = 1.0
    sa_cooling: float = 0.995
    enable_merge_split: bool = False
    merge_split_tries: int = 20


def lns_improve(
    problem: Problem,
    sol0: Solution,
    evaluator: Evaluator,
    params: LNSParams,
    time_budget_s: float = 300.0,
    *,
    rng: Optional[random.Random] = None,
) -> Solution:
    """
    Improve a solution using Large Neighbourhood Search.

    Iteratively destroys and repairs the solution, accepting improvements
    and optionally using simulated annealing.

    Args:
        problem: The problem instance
        sol0: Initial solution
        evaluator: Cost evaluator
        params: LNS parameters
        time_budget_s: Time budget in seconds
        rng: Random number generator

    Returns:
        Improved solution
    """
    if rng is None:
        rng = random.Random()

    n = problem.graph.number_of_nodes()
    gold = [problem.graph.nodes[i].get("gold", 0) for i in range(n)]

    best = sol0.copy()
    best_cost = evaluator.solution_cost(best)
    current = sol0.copy()
    current_cost = best_cost

    start = time.perf_counter()
    iteration = 0
    temperature = params.sa_initial_temp

    while time.perf_counter() - start < time_budget_s:
        if params.max_iter is not None and iteration >= params.max_iter:
            break

        # Destroy phase
        destroyed, removed = _destroy(current, params.destroy_fraction, rng)

        # Repair phase
        repaired = _repair(problem, destroyed, removed, gold, evaluator, params.regret_k, rng)

        # Local search
        repaired = _local_search(problem, repaired, evaluator, rng)

        # Merge/split if enabled
        if params.enable_merge_split:
            repaired = _merge_split(
                problem, repaired, evaluator, rng, tries=params.merge_split_tries
            )
            repaired = _local_search(problem, repaired, evaluator, rng)

        # Evaluate new solution
        new_cost = evaluator.solution_cost(repaired)

        # Acceptance criterion
        accept = False
        if new_cost < current_cost:
            accept = True
        elif params.use_sa and temperature > 1e-6:
            delta = new_cost - current_cost
            if rng.random() < math.exp(-delta / temperature):
                accept = True

        if accept:
            current = repaired
            current_cost = new_cost
            if new_cost < best_cost:
                best = current.copy()
                best_cost = new_cost

        temperature *= params.sa_cooling
        iteration += 1

    return best


def _destroy(
    sol: Solution, fraction: float, rng: random.Random
) -> Tuple[Solution, List[Tuple[int, float]]]:
    """
    Remove a fraction of stops from the solution.

    Args:
        sol: Current solution
        fraction: Fraction of stops to remove
        rng: Random number generator

    Returns:
        Tuple of (destroyed solution, list of removed (node, pickup) pairs)
    """
    removed: List[Tuple[int, float]] = []
    new_trips: List[Trip] = []

    # Collect all removable stops
    all_stops: List[Tuple[int, int, int, float]] = []
    for ti, trip in enumerate(sol.trips):
        for si in range(1, len(trip.stops) - 1):
            node, pick = trip.stops[si], trip.pickups[si]
            if pick > 1e-9 and node != 0:
                all_stops.append((ti, si, node, pick))

    # Select stops to remove
    k = min(max(1, int(len(all_stops) * fraction)), len(all_stops))
    to_remove = set(rng.sample(range(len(all_stops)), k))

    # Build new trips without removed stops
    for ti, trip in enumerate(sol.trips):
        new_stops, new_pickups = [0], [0.0]

        for si in range(1, len(trip.stops) - 1):
            idx = next(
                (i for i, (tii, sii, _, _) in enumerate(all_stops) if tii == ti and sii == si),
                -1,
            )
            if idx >= 0 and idx in to_remove:
                _, _, node, pick = all_stops[idx]
                removed.append((node, pick))
            else:
                new_stops.append(trip.stops[si])
                new_pickups.append(trip.pickups[si])

        new_stops.append(0)
        new_pickups.append(0.0)

        if len(new_stops) > 2:
            new_trips.append(Trip(stops=new_stops, pickups=new_pickups))

    return Solution(trips=new_trips, unserved=sol.unserved), removed


def _repair(
    problem: Problem,
    sol: Solution,
    removed: List[Tuple[int, float]],
    gold: List[float],
    evaluator: Evaluator,
    regret_k: int,
    rng: random.Random,
) -> Solution:
    """
    Reinsert removed stops using regret-k heuristic.

    Args:
        problem: The problem instance
        sol: Solution with removed stops
        removed: List of (node, pickup) pairs to reinsert
        gold: Gold quantities per node
        evaluator: Cost evaluator
        regret_k: Number of best positions to consider
        rng: Random number generator

    Returns:
        Repaired solution
    """
    if not removed:
        return sol

    trips = [t.copy() for t in sol.trips]

    for node, pick in removed:
        # Evaluate all insertion positions
        costs_per_insert: List[Tuple[float, int, int]] = []

        for ti, trip in enumerate(trips):
            for pos in range(1, len(trip.stops)):
                t2 = Trip(
                    stops=trip.stops[:pos] + [node] + trip.stops[pos:],
                    pickups=trip.pickups[:pos] + [pick] + trip.pickups[pos:],
                )
                costs_per_insert.append((evaluator.trip_cost(t2, use_cache=False), ti, pos))

        if not costs_per_insert:
            # Create new trip if no insertion positions
            trips.append(Trip(stops=[0, node, 0], pickups=[0.0, pick, 0.0]))
            continue

        # Select from k-best positions (regret-k)
        costs_per_insert.sort(key=lambda x: x[0])
        if regret_k >= 2 and len(costs_per_insert) >= 2:
            idx = rng.randint(0, min(regret_k, len(costs_per_insert)) - 1)
        else:
            idx = 0

        _, best_ti, best_pos = costs_per_insert[idx]

        # Apply insertion
        t = trips[best_ti]
        t.stops = t.stops[:best_pos] + [node] + t.stops[best_pos:]
        t.pickups = t.pickups[:best_pos] + [pick] + t.pickups[best_pos:]
        t.cached_cost = t.prefix_loads = None

    return Solution(trips=trips, unserved=sol.unserved)


def _local_search(
    problem: Problem,
    sol: Solution,
    evaluator: Evaluator,
    rng: random.Random,
    max_rounds: int = 10,
) -> Solution:
    """
    Apply local search operators: 2-opt, relocate, swap.

    Args:
        problem: The problem instance
        sol: Current solution
        evaluator: Cost evaluator
        rng: Random number generator
        max_rounds: Maximum improvement rounds

    Returns:
        Improved solution
    """
    current = sol.copy()
    cost = evaluator.solution_cost(current)

    improved = True
    for _ in range(max_rounds):
        if not improved:
            break
        improved = False

        current, cost, improved = _two_opt_intra(current, evaluator, cost, improved)
        current, cost, improved = _relocate(current, evaluator, cost, improved)
        current, cost, improved = _swap_intra(current, evaluator, cost, improved)

    return current


def _two_opt_intra(
    sol: Solution, evaluator: Evaluator, cost: float, improved: bool
) -> Tuple[Solution, float, bool]:
    """Apply 2-opt within each trip."""
    for ti, trip in enumerate(sol.trips):
        stops, pickups = trip.stops, trip.pickups
        k = len(stops) - 2
        if k < 2:
            continue

        for i in range(1, k):
            for j in range(i + 1, k + 1):
                t2 = Trip(
                    stops=stops[:i] + list(reversed(stops[i : j + 1])) + stops[j + 1 :],
                    pickups=pickups[:i] + list(reversed(pickups[i : j + 1])) + pickups[j + 1 :],
                )
                new_cost = evaluator.solution_cost(
                    Solution(trips=sol.trips[:ti] + [t2] + sol.trips[ti + 1 :], unserved=sol.unserved)
                )

                if new_cost < cost - 1e-9:
                    new_trips = [t.copy() for t in sol.trips]
                    new_trips[ti] = t2
                    return Solution(trips=new_trips, unserved=sol.unserved), new_cost, True

    return sol, cost, improved


def _relocate(
    sol: Solution, evaluator: Evaluator, cost: float, improved: bool
) -> Tuple[Solution, float, bool]:
    """Try relocating stops between trips."""
    for ti, trip in enumerate(sol.trips):
        if len(trip.stops) <= 3:
            continue

        for si in range(1, len(trip.stops) - 1):
            node, pick = trip.stops[si], trip.pickups[si]

            for tj, trip2 in enumerate(sol.trips):
                for pos in range(1, len(trip2.stops)):
                    if ti == tj and (pos == si or pos == si + 1):
                        continue

                    t1_stops = list(trip.stops)
                    t1_pick = list(trip.pickups)
                    t1_stops.pop(si)
                    t1_pick.pop(si)

                    if len(t1_stops) <= 2:
                        continue

                    pos_adj = pos - 1 if ti == tj and pos > si else pos
                    t2_stops = list(trip2.stops)
                    t2_pick = list(trip2.pickups)
                    t2_stops.insert(pos_adj, node)
                    t2_pick.insert(pos_adj, pick)

                    new_trips = [t.copy() for t in sol.trips]
                    new_trips[ti] = Trip(stops=t1_stops, pickups=t1_pick)
                    new_trips[tj] = Trip(stops=t2_stops, pickups=t2_pick)

                    new_cost = evaluator.solution_cost(
                        Solution(trips=new_trips, unserved=sol.unserved)
                    )

                    if new_cost < cost - 1e-9:
                        return Solution(trips=new_trips, unserved=sol.unserved), new_cost, True

    return sol, cost, improved


def _swap_intra(
    sol: Solution, evaluator: Evaluator, cost: float, improved: bool
) -> Tuple[Solution, float, bool]:
    """Swap positions within trips."""
    for ti, trip in enumerate(sol.trips):
        stops, pickups = list(trip.stops), list(trip.pickups)

        for i in range(1, len(stops) - 1):
            for j in range(i + 1, len(stops) - 1):
                stops[i], stops[j] = stops[j], stops[i]
                pickups[i], pickups[j] = pickups[j], pickups[i]

                t2 = Trip(stops=stops, pickups=pickups)
                new_trips = [t.copy() for t in sol.trips]
                new_trips[ti] = t2

                new_cost = evaluator.solution_cost(
                    Solution(trips=new_trips, unserved=sol.unserved)
                )

                if new_cost < cost - 1e-9:
                    return Solution(trips=new_trips, unserved=sol.unserved), new_cost, True

                # Swap back
                stops[i], stops[j] = stops[j], stops[i]
                pickups[i], pickups[j] = pickups[j], pickups[i]

    return sol, cost, improved


def _merge_two_trips_best_order(
    problem: Problem, evaluator: Evaluator, rng: random.Random, a: Trip, b: Trip
) -> Trip:
    """Merge two trips with best ordering."""
    oracle = evaluator.oracle
    nodes = list(a.stops[1:-1]) + list(b.stops[1:-1])

    if not nodes:
        return Trip(stops=[0, 0], pickups=[0.0, 0.0])

    pickups_map = {node: float(pick) for node, pick in zip(a.stops[1:-1], a.pickups[1:-1])}
    pickups_map.update({node: float(pick) for node, pick in zip(b.stops[1:-1], b.pickups[1:-1])})

    # Build tour using nearest neighbor
    remaining = set(nodes)
    cur = rng.choice(nodes)
    order = [cur]
    remaining.discard(cur)

    while remaining:
        nxt = min(remaining, key=lambda j: oracle.a_hat(cur, j))
        order.append(nxt)
        remaining.discard(nxt)
        cur = nxt

    # Improve with 2-opt
    best = list(order)
    best_cost = evaluator.trip_cost(
        Trip(stops=[0] + best + [0], pickups=[0.0] + [pickups_map[i] for i in best] + [0.0]),
        use_cache=False,
    )

    for _ in range(150):
        improved = False
        n = len(best)

        for i in range(n - 1):
            for j in range(i + 1, n):
                cand = best[:i] + list(reversed(best[i : j + 1])) + best[j + 1 :]
                c = evaluator.trip_cost(
                    Trip(
                        stops=[0] + cand + [0],
                        pickups=[0.0] + [pickups_map[k] for k in cand] + [0.0],
                    ),
                    use_cache=False,
                )

                if c < best_cost - 1e-9:
                    best = cand
                    best_cost = c
                    improved = True
                    break

            if improved:
                break

        if not improved:
            break

    return Trip(
        stops=[0] + best + [0],
        pickups=[0.0] + [pickups_map[i] for i in best] + [0.0],
    )


def _merge_split(
    problem: Problem,
    sol: Solution,
    evaluator: Evaluator,
    rng: random.Random,
    *,
    tries: int = 20,
) -> Solution:
    """Try merging random trip pairs."""
    if len(sol.trips) < 2:
        return sol

    best = sol.copy()
    best_cost = evaluator.solution_cost(best)

    for _ in range(max(1, tries)):
        if len(best.trips) < 2:
            break

        i, j = rng.sample(range(len(best.trips)), 2)
        if i > j:
            i, j = j, i

        ta, tb = best.trips[i], best.trips[j]
        if len(ta.stops) <= 2 or len(tb.stops) <= 2:
            continue

        cost_sep = evaluator.trip_cost(ta) + evaluator.trip_cost(tb)
        merged = _merge_two_trips_best_order(problem, evaluator, rng, ta, tb)

        if evaluator.trip_cost(merged, use_cache=False) < cost_sep - 1e-9:
            new_trips = [t.copy() for t in best.trips]
            new_trips.pop(j)
            new_trips[i] = merged

            cand = Solution(trips=new_trips, unserved=best.unserved)
            cand_cost = evaluator.solution_cost(cand)

            if cand_cost < best_cost - 1e-9:
                best = cand
                best_cost = cand_cost

    return best