"""
Solution evaluator for the Gold Collection Problem.

Provides methods to compute trip and solution costs using either
approximate (oracle-based) or exact (Dijkstra-based) calculations.
"""

from typing import List, Tuple, TYPE_CHECKING

from .problem import Problem
from .solution import Trip, Solution

if TYPE_CHECKING:
    from ..distance.oracle import LandmarkOracle


class Evaluator:
    """
    Evaluates costs of trips and solutions.

    Uses a distance oracle for fast approximate cost calculations during
    heuristic search, with the option for exact cost computation for
    final evaluation.

    Attributes:
        problem: The problem instance
        oracle: Distance oracle for cost approximation
    """

    def __init__(self, problem: Problem, oracle: "LandmarkOracle"):
        """
        Initialize the evaluator.

        Args:
            problem: The problem instance
            oracle: Distance oracle for leg cost approximation
        """
        self.problem = problem
        self.oracle = oracle

    def leg_cost(self, u: int, v: int, w: float) -> float:
        """
        Compute approximate cost of traversing from u to v with load w.

        Uses the oracle for fast approximation.

        Args:
            u: Source node
            v: Destination node
            w: Current load

        Returns:
            Approximate leg cost
        """
        return self.oracle.leg_cost(u, v, w)

    def exact_leg_cost(self, u: int, v: int, w: float) -> float:
        """
        Compute exact cost of traversing from u to v with load w.

        Uses Dijkstra for accurate path cost.

        Args:
            u: Source node
            v: Destination node
            w: Current load

        Returns:
            Exact leg cost
        """
        return self.oracle.exact_leg_cost(u, v, w)

    def trip_cost(self, trip: Trip, use_cache: bool = True) -> float:
        """
        Compute the cost of a trip using oracle approximation.

        Args:
            trip: The trip to evaluate
            use_cache: Whether to use cached cost if available

        Returns:
            Approximate trip cost
        """
        if use_cache and trip.cached_cost is not None and trip.prefix_loads is not None:
            return trip.cached_cost

        cost, prefix = self.recompute_trip_cost(trip, start_index=0)
        trip.cached_cost = cost
        trip.prefix_loads = prefix
        return cost

    def recompute_trip_cost(
        self, trip: Trip, start_index: int = 0
    ) -> Tuple[float, List[float]]:
        """
        Recompute trip cost from a given starting index.

        Args:
            trip: The trip to evaluate
            start_index: Index to start cost computation from

        Returns:
            Tuple of (cost, prefix_loads)
        """
        stops = trip.stops
        pickups = trip.pickups
        n = len(stops)

        if n <= 1:
            return 0.0, list(pickups)

        # Compute prefix loads (cumulative gold at each stop)
        prefix_loads: List[float] = [0.0] * n
        prefix_loads[0] = pickups[0]
        for i in range(1, n):
            prefix_loads[i] = prefix_loads[i - 1] + pickups[i]

        # Sum leg costs
        cost = 0.0
        for i in range(start_index, n - 1):
            load = prefix_loads[i]
            cost += self.leg_cost(stops[i], stops[i + 1], load)

        return cost, prefix_loads

    def solution_cost(self, sol: Solution) -> float:
        """
        Compute the total cost of a solution using oracle approximation.

        Args:
            sol: The solution to evaluate

        Returns:
            Approximate total cost
        """
        return sum(self.trip_cost(t, use_cache=True) for t in sol.trips)

    def exact_solution_cost(self, sol: Solution) -> float:
        """
        Compute the exact cost of a solution using Dijkstra.

        Args:
            sol: The solution to evaluate

        Returns:
            Exact total cost
        """
        total = 0.0
        for trip in sol.trips:
            stops = trip.stops
            pickups = trip.pickups
            n = len(stops)
            load = 0.0
            for i in range(n - 1):
                total += self.exact_leg_cost(stops[i], stops[i + 1], load)
                load += pickups[i + 1]
        return total

    def delta_swap(self, trip: Trip, i: int, j: int) -> float:
        """
        Compute cost change from swapping positions i and j in a trip.

        Args:
            trip: The trip
            i: First position index
            j: Second position index

        Returns:
            Cost difference (positive = worse)
        """
        if i == j or i < 0 or j < 0 or i >= len(trip.stops) or j >= len(trip.stops):
            return 0.0

        before = self.trip_cost(trip, use_cache=True)

        # Create modified trip
        stops = list(trip.stops)
        pickups = list(trip.pickups)
        stops[i], stops[j] = stops[j], stops[i]
        pickups[i], pickups[j] = pickups[j], pickups[i]

        temp_trip = Trip(stops=stops, pickups=pickups)
        after = self.trip_cost(temp_trip, use_cache=False)

        return after - before

    def insertion_cost(
        self, trip: Trip, position: int, node: int, pickup: float
    ) -> float:
        """
        Compute cost of inserting a node at given position.

        Args:
            trip: The trip
            position: Position to insert (1 to len-1)
            node: Node to insert
            pickup: Gold pickup amount

        Returns:
            New trip cost after insertion
        """
        new_stops = trip.stops[:position] + [node] + trip.stops[position:]
        new_pickups = trip.pickups[:position] + [pickup] + trip.pickups[position:]
        temp_trip = Trip(stops=new_stops, pickups=new_pickups)
        return self.trip_cost(temp_trip, use_cache=False)

    def __repr__(self) -> str:
        return f"Evaluator(problem={self.problem})"