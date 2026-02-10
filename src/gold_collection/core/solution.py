"""
Solution representation for the Gold Collection Problem.

A solution consists of one or more trips, where each trip:
- Starts at depot (node 0)
- Visits a sequence of cities, collecting gold
- Returns to depot
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict

from .problem import Problem


@dataclass
class Trip:
    """
    Represents a single trip from depot and back.

    A trip is a sequence of stops where:
    - First stop is always depot (0) with pickup 0
    - Last stop is always depot (0) with pickup 0
    - Intermediate stops are cities with their gold pickups

    Attributes:
        stops: List of node indices [0, city1, city2, ..., 0]
        pickups: List of gold amounts picked up at each stop
        cached_cost: Memoized trip cost (invalidated on modification)
        prefix_loads: Cumulative loads at each stop
    """

    stops: List[int]
    pickups: List[float]
    cached_cost: Optional[float] = None
    prefix_loads: Optional[List[float]] = None

    def __post_init__(self) -> None:
        """Validate that stops and pickups have matching lengths."""
        if len(self.stops) != len(self.pickups):
            raise ValueError(
                f"stops ({len(self.stops)}) and pickups ({len(self.pickups)}) "
                "must have the same length"
            )

    def copy(self) -> "Trip":
        """Create a deep copy of this trip."""
        return Trip(
            stops=list(self.stops),
            pickups=list(self.pickups),
            cached_cost=self.cached_cost,
            prefix_loads=list(self.prefix_loads) if self.prefix_loads else None,
        )

    def invalidate_cache(self) -> None:
        """Invalidate cached cost and prefix loads."""
        self.cached_cost = None
        self.prefix_loads = None

    @property
    def num_stops(self) -> int:
        """Number of stops including depot visits."""
        return len(self.stops)

    @property
    def num_customers(self) -> int:
        """Number of customer stops (excluding depot)."""
        return max(0, len(self.stops) - 2)

    @property
    def total_pickup(self) -> float:
        """Total gold collected in this trip."""
        return sum(self.pickups)

    def get_customers(self) -> List[int]:
        """Get list of customer nodes (excluding depot)."""
        return [s for s in self.stops[1:-1] if s != 0]

    def is_valid(self) -> bool:
        """Check if trip structure is valid."""
        if len(self.stops) < 2:
            return False
        if self.stops[0] != 0 or self.stops[-1] != 0:
            return False
        if any(p < 0 for p in self.pickups):
            return False
        return True

    def __repr__(self) -> str:
        customers = self.get_customers()
        total = self.total_pickup
        return f"Trip({len(customers)} customers, {total:.1f} gold)"


@dataclass
class Solution:
    """
    Represents a complete solution to the Gold Collection Problem.

    A solution is a collection of trips that together collect all gold
    from all cities and return it to the depot.

    Attributes:
        trips: List of Trip objects
        unserved: Dictionary of {city: remaining_gold} for any unserved demand
    """

    trips: List[Trip] = field(default_factory=list)
    unserved: Dict[int, float] = field(default_factory=dict)

    def copy(self) -> "Solution":
        """Create a deep copy of this solution."""
        return Solution(
            trips=[t.copy() for t in self.trips],
            unserved=dict(self.unserved),
        )

    @property
    def num_trips(self) -> int:
        """Number of trips in the solution."""
        return len(self.trips)

    @property
    def total_customers_served(self) -> int:
        """Total number of customer visits across all trips."""
        return sum(t.num_customers for t in self.trips)

    @property
    def total_gold_collected(self) -> float:
        """Total gold collected across all trips."""
        return sum(t.total_pickup for t in self.trips)

    def is_feasible(self, problem: Problem) -> bool:
        """
        Check if the solution is feasible for the given problem.

        A solution is feasible if:
        - All trips start and end at depot (node 0)
        - All nodes are valid (exist in problem)
        - Pickups don't exceed available gold at each city
        - All gold is collected (nothing unserved)

        Args:
            problem: The problem instance to check against

        Returns:
            True if solution is feasible, False otherwise
        """
        G = problem.graph
        n = G.number_of_nodes()

        # Get gold quantities
        gold = {i: G.nodes[i].get("gold", 0) for i in range(n)}
        remaining = dict(gold)

        # Validate each trip
        for trip in self.trips:
            if not trip.stops:
                continue

            # Check depot start/end
            if trip.stops[0] != 0 or trip.stops[-1] != 0:
                return False

            # Check valid nodes
            for node_id in trip.stops:
                if node_id < 0 or node_id >= n:
                    return False

            # Check pickups don't exceed available gold
            for stop, pickup in zip(trip.stops, trip.pickups):
                if pickup < 0:
                    return False
                if stop != 0 and stop in remaining:
                    if pickup > remaining[stop] + 1e-9:
                        return False
                    remaining[stop] -= pickup

        # Check all gold is collected
        for i in range(1, n):
            if remaining.get(i, 0) > 1e-9:
                return False

        # Check no unserved demand
        if self.unserved:
            for v in self.unserved.values():
                if v > 1e-9:
                    return False

        return True

    def get_trip_summary(self) -> str:
        """Get a summary string of all trips."""
        lines = [f"Solution with {self.num_trips} trips:"]
        for i, trip in enumerate(self.trips):
            lines.append(f"  Trip {i+1}: {trip}")
        if self.unserved:
            unserved_gold = sum(self.unserved.values())
            lines.append(f"  Unserved: {unserved_gold:.1f} gold")
        return "\n".join(lines)

    def __repr__(self) -> str:
        return (
            f"Solution(trips={self.num_trips}, "
            f"gold={self.total_gold_collected:.1f})"
        )