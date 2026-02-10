"""
Problem definition for the Gold Collection Problem.

The Gold Collection Problem is a variant of the Vehicle Routing Problem (VRP)
where the cost of traversing an edge depends on the current load being carried.
"""

from itertools import combinations
from typing import Tuple

import numpy as np
import networkx as nx


class Problem:
    """
    Represents a Gold Collection Problem instance.

    The problem consists of a graph where:
    - Node 0 is the depot
    - Each other node i has a gold quantity g_i >= 0
    - Each edge (u, v) has a distance d(u, v)

    The cost of traversing edge (u, v) with load w is:
        C = d(u,v) + (alpha * d(u,v) * w)^beta

    Attributes:
        alpha: Scale parameter for load penalty (>= 0)
        beta: Non-linearity parameter (>= 0)
            - beta < 1: Sub-linear (combining pickups reduces cost)
            - beta = 1: Linear (classic VRP behavior)
            - beta > 1: Super-linear (many small trips preferred)
    """

    def __init__(
        self,
        num_cities: int,
        *,
        alpha: float = 1.0,
        beta: float = 1.0,
        density: float = 0.5,
        seed: int = 42,
    ):
        """
        Initialize a Gold Collection Problem instance.

        Args:
            num_cities: Total number of nodes (including depot at index 0)
            alpha: Load penalty scale parameter
            beta: Load penalty non-linearity parameter
            density: Edge probability for random graph generation
            seed: Random seed for reproducibility
        """
        self._validate_parameters(num_cities, alpha, beta, density)

        rng = np.random.default_rng(seed)
        self._alpha = alpha
        self._beta = beta
        self._seed = seed
        self._density = density

        # Generate random city positions
        cities = rng.random(size=(num_cities, 2))
        cities[0, 0] = cities[0, 1] = 0.5  # Depot at center

        # Build graph
        self._graph = nx.Graph()
        self._add_nodes(cities, rng)
        self._add_edges(cities, rng, density)

        assert nx.is_connected(self._graph), "Generated graph must be connected"

    @staticmethod
    def _validate_parameters(
        num_cities: int, alpha: float, beta: float, density: float
    ) -> None:
        """Validate input parameters."""
        if num_cities < 1:
            raise ValueError("num_cities must be at least 1")
        if alpha < 0:
            raise ValueError("alpha must be non-negative")
        if beta < 0:
            raise ValueError("beta must be non-negative")
        if not 0 < density <= 1:
            raise ValueError("density must be in (0, 1]")

    def _add_nodes(self, cities: np.ndarray, rng: np.random.Generator) -> None:
        """Add nodes with positions and gold quantities."""
        num_cities = len(cities)
        # Depot has no gold
        self._graph.add_node(0, pos=(cities[0, 0], cities[0, 1]), gold=0)
        # Other cities have random gold [1, 1000]
        for i in range(1, num_cities):
            gold_amount = 1 + 999 * rng.random()
            self._graph.add_node(i, pos=(cities[i, 0], cities[i, 1]), gold=gold_amount)

    def _add_edges(
        self, cities: np.ndarray, rng: np.random.Generator, density: float
    ) -> None:
        """Add edges based on Euclidean distances."""
        num_cities = len(cities)
        # Compute pairwise distances
        diff = cities[:, np.newaxis, :] - cities[np.newaxis, :, :]
        distances = np.sqrt(np.sum(np.square(diff), axis=-1))

        for c1, c2 in combinations(range(num_cities), 2):
            # Add edge with probability density, or always if consecutive
            if rng.random() < density or c2 == c1 + 1:
                self._graph.add_edge(c1, c2, dist=distances[c1, c2])

    @property
    def graph(self) -> nx.Graph:
        """Return a copy of the underlying graph."""
        return nx.Graph(self._graph)

    @property
    def alpha(self) -> float:
        """Load penalty scale parameter."""
        return self._alpha

    @property
    def beta(self) -> float:
        """Load penalty non-linearity parameter."""
        return self._beta

    @property
    def num_cities(self) -> int:
        """Total number of nodes including depot."""
        return self._graph.number_of_nodes()

    @property
    def total_gold(self) -> float:
        """Total gold to collect across all cities."""
        return sum(
            self._graph.nodes[i].get("gold", 0)
            for i in range(1, self.num_cities)
        )

    def get_gold(self, node: int) -> float:
        """Get gold quantity at a node."""
        return self._graph.nodes[node].get("gold", 0)

    def get_position(self, node: int) -> Tuple[float, float]:
        """Get (x, y) position of a node."""
        pos = self._graph.nodes[node].get("pos", (0.0, 0.0))
        return (pos[0], pos[1])

    def edge_cost(self, path: list, weight: float) -> float:
        """
        Compute the cost of traversing a path with given load.

        Args:
            path: List of nodes forming the path
            weight: Current load being carried

        Returns:
            Total cost = distance + (alpha * distance * weight)^beta
        """
        dist = nx.path_weight(self._graph, path, weight="dist")
        return dist + (self._alpha * dist * weight) ** self._beta

    def baseline(self) -> float:
        """
        Compute baseline cost using naive policy.

        The naive policy visits each city independently:
        for each city i, do a round-trip 0 -> i -> 0.

        Returns:
            Total cost of the baseline solution
        """
        total_cost = 0.0
        paths = nx.single_source_dijkstra_path(self._graph, source=0, weight="dist")

        for dest, path in paths.items():
            if dest == 0:
                continue
            gold = self._graph.nodes[dest]["gold"]
            if gold <= 0:
                continue

            # Cost of each leg in round-trip
            for c1, c2 in zip(path, path[1:]):
                # Outbound: no load
                total_cost += self.edge_cost([c1, c2], 0)
                # Return: carrying gold
                total_cost += self.edge_cost([c1, c2], gold)

        return total_cost

    def __repr__(self) -> str:
        return (
            f"Problem(n={self.num_cities}, alpha={self._alpha}, "
            f"beta={self._beta}, density={self._density}, seed={self._seed})"
        )