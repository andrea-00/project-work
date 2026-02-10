"""
Constructive heuristics for the Gold Collection Problem.

Provides regime-specific construction methods:
- Regime T (beta < 1): Giant tour + split DP
- Regime L (beta = 1): Star-based with Clarke-Wright merging
- Regime S (beta > 1): Soft-capacity chunking
"""

import random
from typing import List, Optional

import numpy as np

from ...core.problem import Problem
from ...core.solution import Trip, Solution
from ...core.evaluator import Evaluator
from ...distance.oracle import LandmarkOracle


def _trip_cost_segment(
    evaluator: Evaluator, tour: List[int], gold: List[float], i: int, j: int
) -> float:
    """Compute cost of a trip covering tour segment [i, j]."""
    segment = tour[i : j + 1]
    stops = [0] + segment + [0]
    pickups = [0.0] + [gold[v] for v in segment] + [0.0]
    trip = Trip(stops=stops, pickups=pickups)
    return evaluator.trip_cost(trip, use_cache=False)


def split_tour(
    problem: Problem,
    tour: List[int],
    evaluator: Evaluator,
    window_W: Optional[int] = None,
) -> Solution:
    """
    Split a giant tour into optimal trips using dynamic programming.

    Given a sequence of cities to visit, finds the optimal partition
    into separate trips that minimizes total cost.

    Args:
        problem: The problem instance
        tour: List of city indices (not including depot)
        evaluator: Cost evaluator
        window_W: Maximum segment length to consider (limits DP states)

    Returns:
        Solution with optimal trip partitioning
    """
    G = problem.graph
    n = G.number_of_nodes()
    gold = [G.nodes[i].get("gold", 0) for i in range(n)]
    N = len(tour)

    if N == 0:
        return Solution(trips=[], unserved={})

    # Default window size based on problem size
    if window_W is None:
        if n <= 500:
            window_W = N
        elif n <= 2000:
            window_W = min(800, N)
        else:
            window_W = min(500, N)

    # DP: dp[j] = min cost to serve tour[0:j]
    dp: List[float] = [float("inf")] * (N + 1)
    pred: List[int] = [-1] * (N + 1)
    dp[0] = 0.0

    for j in range(N):
        start = max(0, j - window_W + 1)
        for i in range(start, j + 1):
            seg_cost = _trip_cost_segment(evaluator, tour, gold, i, j)
            cand = dp[i] + seg_cost
            if cand < dp[j + 1]:
                dp[j + 1] = cand
                pred[j + 1] = i

    # Reconstruct trips from predecessors
    trips: List[Trip] = []
    curr = N
    while curr > 0:
        prev = pred[curr]
        segment = tour[prev:curr]
        stops = [0] + segment + [0]
        pickups = [0.0] + [gold[v] for v in segment] + [0.0]
        trips.append(Trip(stops=stops, pickups=pickups))
        curr = prev

    trips.reverse()
    return Solution(trips=trips, unserved={})


def construct_tour_regime_t(
    problem: Problem,
    oracle: LandmarkOracle,
    *,
    rng: Optional[random.Random] = None,
) -> List[int]:
    """
    Construct a giant tour for Regime T (sub-linear costs).

    Uses nearest insertion followed by 2-opt improvement.

    Args:
        problem: The problem instance
        oracle: Distance oracle for cost estimation
        rng: Random number generator

    Returns:
        List of city indices forming the tour
    """
    G = problem.graph
    n = G.number_of_nodes()
    nodes = list(range(1, n))

    if not nodes:
        return []

    if rng is None:
        rng = random.Random()

    def dist(u: int, v: int) -> float:
        return oracle.a_hat(u, v)

    tour = _nearest_insertion(nodes, dist, rng)
    tour = _two_opt_tour(tour, dist)
    return tour


def _nearest_insertion(
    nodes: List[int], dist: callable, rng: random.Random
) -> List[int]:
    """Build tour using nearest insertion heuristic."""
    if len(nodes) <= 1:
        return list(nodes)

    unvisited = set(nodes)
    start = rng.choice(nodes)
    tour = [start]
    unvisited.discard(start)

    while unvisited:
        best_c = None
        best_pos = 0
        best_inc = float("inf")

        for c in unvisited:
            for pos in range(len(tour) + 1):
                if pos == 0:
                    inc = dist(0, c) + dist(c, tour[0]) - dist(0, tour[0])
                elif pos == len(tour):
                    inc = dist(tour[-1], c) + dist(c, 0) - dist(tour[-1], 0)
                else:
                    inc = (
                        dist(tour[pos - 1], c)
                        + dist(c, tour[pos])
                        - dist(tour[pos - 1], tour[pos])
                    )
                if inc < best_inc:
                    best_inc = inc
                    best_c = c
                    best_pos = pos

        if best_c is None:
            break

        tour.insert(best_pos, best_c)
        unvisited.discard(best_c)

    return tour


def _two_opt_tour(tour: List[int], dist: callable) -> List[int]:
    """Improve tour using 2-opt moves."""
    n = len(tour)
    if n <= 2:
        return tour

    improved = True
    while improved:
        improved = False
        for i in range(n - 1):
            for j in range(i + 2, n + 1):
                if j == n and i == 0:
                    continue

                a, b = tour[i], tour[(i + 1) % n]
                c, d = tour[(j - 1) % n], tour[j % n]
                before = dist(a, b) + dist(c, d)
                after = dist(a, c) + dist(b, d)

                if after < before - 1e-9:
                    tour[i + 1 : j] = reversed(tour[i + 1 : j])
                    improved = True
                    break

            if improved:
                break

    return tour


def _nn_order(
    problem: Problem, oracle: LandmarkOracle, nodes: List[int], rng: random.Random
) -> List[int]:
    """Order nodes using nearest neighbor heuristic."""
    if not nodes:
        return []

    remaining = set(nodes)
    cur = rng.choice(nodes)
    order = [cur]
    remaining.discard(cur)

    while remaining:
        nxt = min(remaining, key=lambda j: oracle.a_hat(cur, j))
        order.append(nxt)
        remaining.discard(nxt)
        cur = nxt

    return order


def _two_opt_customers(
    problem: Problem, evaluator: Evaluator, customers: List[int], pickups_map: dict
) -> Trip:
    """Create trip from customers and improve with 2-opt."""
    if len(customers) <= 1:
        stops = [0] + customers + [0]
        picks = [0.0] + [float(pickups_map.get(i, 0)) for i in customers] + [0.0]
        return Trip(stops=stops, pickups=picks)

    best = list(customers)
    best_trip = Trip(
        stops=[0] + best + [0],
        pickups=[0.0] + [float(pickups_map.get(i, 0)) for i in best] + [0.0],
    )
    best_cost = evaluator.trip_cost(best_trip, use_cache=False)

    improved = True
    for _ in range(200):
        if not improved:
            break
        improved = False
        n = len(best)

        for i in range(n - 1):
            for j in range(i + 1, n):
                cand = best[:i] + list(reversed(best[i : j + 1])) + best[j + 1 :]
                t2 = Trip(
                    stops=[0] + cand + [0],
                    pickups=[0.0]
                    + [float(pickups_map.get(k, 0)) for k in cand]
                    + [0.0],
                )
                c = evaluator.trip_cost(t2, use_cache=False)

                if c < best_cost - 1e-9:
                    best = cand
                    best_cost = c
                    improved = True
                    break

            if improved:
                break

    return Trip(
        stops=[0] + best + [0],
        pickups=[0.0] + [float(pickups_map.get(i, 0)) for i in best] + [0.0],
    )


def _merge_trips_resequence(
    problem: Problem,
    evaluator: Evaluator,
    oracle: LandmarkOracle,
    ta: Trip,
    tb: Trip,
    rng: random.Random,
) -> Trip:
    """Merge two trips and find best sequence with NN + 2-opt."""
    a_nodes = ta.stops[1:-1]
    b_nodes = tb.stops[1:-1]
    nodes = list(a_nodes) + list(b_nodes)

    pickups_map = {}
    for node, pick in zip(ta.stops[1:-1], ta.pickups[1:-1]):
        pickups_map[node] = float(pick)
    for node, pick in zip(tb.stops[1:-1], tb.pickups[1:-1]):
        pickups_map[node] = float(pick)

    order = _nn_order(problem, oracle, nodes, rng)
    return _two_opt_customers(problem, evaluator, order, pickups_map)


def construct_solution_regime_l(
    problem: Problem,
    evaluator: Evaluator,
    *,
    max_trip_stops: Optional[int] = None,
    rng: Optional[random.Random] = None,
) -> Solution:
    """
    Construct solution for Regime L (linear costs).

    Uses star-based construction with Clarke-Wright style merging.

    Args:
        problem: The problem instance
        evaluator: Cost evaluator
        max_trip_stops: Maximum stops per trip
        rng: Random number generator

    Returns:
        Constructed solution
    """
    G = problem.graph
    n = G.number_of_nodes()
    gold = [G.nodes[i].get("gold", 0) for i in range(n)]
    oracle = evaluator.oracle

    if rng is None:
        rng = random.Random()

    # Start with individual trips for each city
    trips: List[Trip] = []
    for i in range(1, n):
        if gold[i] <= 0:
            continue
        trips.append(Trip(stops=[0, i, 0], pickups=[0.0, float(gold[i]), 0.0]))

    max_trip_stops = max_trip_stops or max(n, 50)

    # Merge trips using savings heuristic
    while True:
        best_saving = -1e9
        best_ia = -1
        best_ib = -1
        best_merged: Optional[Trip] = None

        for ia, ta in enumerate(trips):
            for ib, tb in enumerate(trips):
                if ia >= ib:
                    continue

                cost_sep = evaluator.trip_cost(ta) + evaluator.trip_cost(tb)
                merged = _merge_trips_resequence(problem, evaluator, oracle, ta, tb, rng)
                cost_m = evaluator.trip_cost(merged, use_cache=False)

                if cost_m < cost_sep and (len(merged.stops) - 2) <= max_trip_stops:
                    saving = cost_sep - cost_m
                    if saving > best_saving:
                        best_saving = saving
                        best_ia, best_ib = ia, ib
                        best_merged = merged

        if best_ia < 0 or best_merged is None:
            break

        # Apply best merge
        for idx in sorted([best_ia, best_ib], reverse=True):
            trips.pop(idx)
        trips.append(best_merged)

    sol = Solution(trips=trips, unserved={})
    sol = _local_improve_regime_l(sol, problem, evaluator)
    return sol


def _local_improve_regime_l(
    sol: Solution,
    problem: Problem,
    evaluator: Evaluator,
    max_2opt_iter: int = 100,
    max_relocate_passes: int = 5,
) -> Solution:
    """Apply local improvement for Regime L (2-opt + relocate)."""
    trips = [t.copy() for t in sol.trips]

    # 2-opt within trips
    for _ in range(max_2opt_iter):
        improved = False
        for ti, trip in enumerate(trips):
            s, p = trip.stops, trip.pickups
            if len(s) <= 3:
                continue

            for i in range(1, len(s) - 1):
                for k in range(i + 1, len(s) - 1):
                    ns = s[:i] + list(reversed(s[i : k + 1])) + s[k + 1 :]
                    np_ = p[:i] + list(reversed(p[i : k + 1])) + p[k + 1 :]
                    t2 = Trip(stops=ns, pickups=np_)
                    new_trips = trips[:ti] + [t2] + trips[ti + 1 :]

                    old_cost = evaluator.solution_cost(
                        Solution(trips=trips, unserved={})
                    )
                    new_cost = evaluator.solution_cost(
                        Solution(trips=new_trips, unserved={})
                    )

                    if new_cost + 1e-12 < old_cost:
                        trips = new_trips
                        improved = True
                        break

                if improved:
                    break

            if improved:
                break

        if not improved:
            break

    # Relocate between trips
    for _ in range(max_relocate_passes):
        improved = False

        for ti, trip in enumerate(trips):
            if len(trip.stops) <= 3:
                continue

            for si in range(1, len(trip.stops) - 1):
                node, pick = trip.stops[si], trip.pickups[si]
                if node == 0:
                    continue

                for tj, trip2 in enumerate(trips):
                    if ti == tj:
                        continue

                    for pos in range(1, len(trip2.stops)):
                        t1_stops = trip.stops[:si] + trip.stops[si + 1 :]
                        t1_pick = trip.pickups[:si] + trip.pickups[si + 1 :]
                        t2_stops = trip2.stops[:pos] + [node] + trip2.stops[pos:]
                        t2_pick = trip2.pickups[:pos] + [pick] + trip2.pickups[pos:]

                        if len(t1_stops) <= 2:
                            continue

                        t_new_i = Trip(stops=t1_stops, pickups=t1_pick)
                        t_new_j = Trip(stops=t2_stops, pickups=t2_pick)
                        full = [
                            t_new_i
                            if idx == ti
                            else t_new_j
                            if idx == tj
                            else t
                            for idx, t in enumerate(trips)
                        ]

                        old_cost = evaluator.solution_cost(
                            Solution(trips=trips, unserved={})
                        )
                        new_cost = evaluator.solution_cost(
                            Solution(trips=full, unserved={})
                        )

                        if new_cost + 1e-12 < old_cost:
                            trips = full
                            improved = True
                            break

                    if improved:
                        break

                if improved:
                    break

            if improved:
                break

        if not improved:
            break

    return Solution(trips=trips, unserved={})


def construct_solution_regime_s(
    problem: Problem,
    evaluator: Evaluator,
    *,
    q_percentile: float = 30.0,
    rng: Optional[random.Random] = None,
) -> Solution:
    """
    Construct solution for Regime S (super-linear costs).

    Uses soft-capacity chunking to create many small trips.

    Args:
        problem: The problem instance
        evaluator: Cost evaluator
        q_percentile: Percentile for computing soft capacity Q
        rng: Random number generator

    Returns:
        Constructed solution
    """
    G = problem.graph
    n = G.number_of_nodes()
    gold = [G.nodes[i].get("gold", 0) for i in range(n)]
    oracle = evaluator.oracle
    alpha = problem.alpha
    beta = problem.beta

    if rng is None:
        rng = random.Random()

    G_total = sum(gold[1:])
    if G_total <= 0:
        return Solution(trips=[], unserved={})

    # Compute optimal capacity Q for each node
    Qi_list: List[float] = []
    for i in range(1, n):
        if gold[i] <= 0:
            continue

        Ai = oracle.a0(i)
        Bi = oracle.b0(i)

        if Bi <= 0 or beta <= 1:
            Qi_list.append(float("inf"))
        else:
            val = (1.0 / alpha) * ((2 * Ai) / ((beta - 1) * Bi)) ** (1.0 / beta)
            Qi_list.append(val if np.isfinite(val) else float("inf"))

    # Use percentile of finite Qi values
    finite_Qi = [q for q in Qi_list if np.isfinite(q)]
    Q = float(np.percentile(finite_Qi or [1.0], q_percentile))

    avg_gold = G_total / max(1, len([i for i in range(1, n) if gold[i] > 0]))
    Q = max(10.0, avg_gold * 0.2, min(Q, G_total))

    # Build trips by chunking
    remaining = dict((i, gold[i]) for i in range(1, n) if gold[i] > 0)
    trips: List[Trip] = []
    max_routes = max(n * 2, 500)

    for _ in range(max_routes):
        if not any(remaining.get(i, 0) > 1e-9 for i in remaining):
            break

        route_stops: List[int] = [0]
        route_pickups: List[float] = [0.0]
        load = 0.0
        unvisited = {i for i in remaining if remaining.get(i, 0) > 1e-9}

        while unvisited and load < Q * 1.1:
            best = None
            best_inc = float("inf")
            current = route_stops[-1]

            for i in unvisited:
                if remaining[i] <= 0:
                    continue

                take = min(remaining[i], Q - load) if load < Q else min(remaining[i], Q * 0.5)
                if take <= 0:
                    continue

                d = oracle.a_hat(current, i) + oracle.a_hat(i, 0)
                if remaining[i] > 0.5 * Q:
                    d *= 1.2

                if d < best_inc:
                    best_inc = d
                    best = i

            if best is None:
                fallback = min(unvisited, key=lambda i: remaining.get(i, 0)) if unvisited else None
                if fallback is not None and remaining.get(fallback, 0) > 1e-9:
                    best = fallback
                else:
                    break

            if best is None:
                break

            take = (
                min(remaining[best], max(1.0, Q - load))
                if load < Q
                else min(remaining[best], max(1.0, Q * 0.5))
            )

            if take <= 0:
                unvisited.discard(best)
                continue

            route_stops.append(best)
            route_pickups.append(take)
            remaining[best] -= take
            load += take

            if remaining[best] < 1e-9:
                unvisited.discard(best)

        route_stops.append(0)
        route_pickups.append(0.0)

        if len(route_stops) > 2:
            trips.append(Trip(stops=route_stops, pickups=route_pickups))

    # Handle any remaining gold
    for i in list(remaining):
        if remaining[i] > 1e-9:
            trips.append(
                Trip(stops=[0, i, 0], pickups=[0.0, remaining[i], 0.0])
            )
            remaining[i] = 0

    return Solution(trips=trips, unserved={k: v for k, v in remaining.items() if v > 1e-9})