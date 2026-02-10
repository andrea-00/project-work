"""
Landmark-based distance oracle for fast cost approximation.

Uses farthest-point sampling to select landmarks and precomputes
distance matrices for efficient lower-bound cost estimation.
"""

import math
import random
from dataclasses import dataclass
from typing import Dict, List, Literal, Optional, Tuple

import networkx as nx
import numpy as np

from ..core.problem import Problem


@dataclass
class OraclePlan:
    """
    Configuration plan for the distance oracle.

    Determined by analyzing the problem graph characteristics.

    Attributes:
        L: Number of landmarks to use
        mode: 'full' for small graphs, 'landmarks' for large
        n: Number of nodes
        m: Number of edges
        deg_avg: Average node degree
        bottleneck_proxy: Measure of graph bottleneck-ness
        euclidean_proxy: How close graph distances are to Euclidean
    """

    L: int
    mode: Literal["full", "landmarks"]
    n: int
    m: int
    deg_avg: float
    bottleneck_proxy: float = 0.0
    euclidean_proxy: float = 1.0


def profile(problem: Problem, *, k_sample: int = 50) -> OraclePlan:
    """
    Analyze problem graph and determine oracle configuration.

    Args:
        problem: The problem instance
        k_sample: Number of samples for euclidean proxy estimation

    Returns:
        OraclePlan with recommended configuration
    """
    G = problem.graph
    n = G.number_of_nodes()
    m = G.number_of_edges()
    deg_avg = (2 * m / n) if n > 0 else 0.0

    # Validate connectivity
    try:
        _ = nx.single_source_dijkstra_path_length(G, 0, weight="dist")
    except Exception:
        pass

    # Estimate bottleneck via edge betweenness
    bottleneck_proxy = 0.0
    if n > 2 and m > 0:
        try:
            betweenness = nx.edge_betweenness_centrality(G, weight="dist")
            bottleneck_proxy = max(betweenness.values()) if betweenness else 0.0
        except Exception:
            pass

    # Estimate how close to Euclidean the graph is
    euclidean_proxy = _estimate_euclidean_proxy(G, n, k_sample)

    # Determine number of landmarks
    L = min(64, max(16, round(math.sqrt(n)))) if n > 0 else 16

    # Use full mode for small/dense graphs
    use_full = n <= 800 or deg_avg >= 10.0
    mode: Literal["full", "landmarks"] = "full" if use_full else "landmarks"

    return OraclePlan(
        L=L,
        mode=mode,
        n=n,
        m=m,
        deg_avg=deg_avg,
        bottleneck_proxy=bottleneck_proxy,
        euclidean_proxy=euclidean_proxy,
    )


def _estimate_euclidean_proxy(G: nx.Graph, n: int, k_sample: int) -> float:
    """Estimate how close graph distances are to direct edge distances."""
    if n < 3:
        return 1.0

    nodes = list(G.nodes())
    sample_size = min(k_sample, (n * (n - 1)) // 2)
    ratios = []

    for _ in range(sample_size):
        u, v = random.sample(nodes, 2)
        if u == v:
            continue
        try:
            path_len = nx.shortest_path_length(G, u, v, weight="dist")
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            continue

        direct = G.get_edge_data(u, v)
        direct_d = direct.get("dist", float("inf")) if direct else float("inf")

        if path_len > 1e-9 and direct_d < float("inf"):
            ratios.append(direct_d / path_len)

    return float(sum(ratios) / len(ratios)) if ratios else 1.0


class LandmarkOracle:
    """
    Distance oracle using landmark-based approximation.

    Provides fast lower-bound estimates for leg costs during
    heuristic search, with exact computation available when needed.

    The oracle precomputes two distance matrices per landmark:
    - A: Standard shortest path distances
    - B: Distances with edge weights d^beta (for load penalty term)

    Leg cost approximation: a_hat(u,v) + ((alpha * w)^beta) * b_hat(u,v)
    """

    def __init__(
        self,
        problem: Problem,
        beta: float,
        L: Optional[int] = None,
        plan: Optional[OraclePlan] = None,
    ):
        """
        Initialize the landmark oracle.

        Args:
            problem: The problem instance
            beta: Non-linearity parameter for load penalty
            L: Number of landmarks (overrides plan if provided)
            plan: Oracle configuration plan
        """
        self.problem = problem
        self.beta = beta
        self.alpha = problem.alpha

        G = problem.graph
        self.n = G.number_of_nodes()

        if plan is None:
            plan = profile(problem)
        self.plan = plan

        num_landmarks = L if L is not None else plan.L
        landmarks = self._farthest_point_sampling(G, num_landmarks)
        self.landmarks = landmarks

        # Precompute distance matrices
        self._A: List[Dict[int, float]] = []
        self._B: List[Dict[int, float]] = []

        for ell in landmarks:
            # Standard distances
            dist_a = dict(nx.single_source_dijkstra_path_length(G, ell, weight="dist"))
            self._A.append(dist_a)

            # Distances with d^beta weights
            def weight_b(u, v, d):
                return (d.get("dist", 1.0)) ** beta

            dist_b = dict(nx.single_source_dijkstra_path_length(G, ell, weight=weight_b))
            self._B.append(dist_b)

        # Precompute depot distances
        self._A0 = np.full(self.n, np.inf, dtype=float)
        self._B0 = np.full(self.n, np.inf, dtype=float)
        for i in range(self.n):
            self._A0[i] = self.a_hat(0, i)
            self._B0[i] = self.b_hat(0, i)
        self._A0[0] = 0.0
        self._B0[0] = 0.0

        # Cache for exact costs
        self._exact_cache: Dict[Tuple[int, int, int], float] = {}
        self._bucket_step = 50.0

    def _farthest_point_sampling(self, G: nx.Graph, L: int) -> List[int]:
        """
        Select landmarks using farthest-point sampling.

        Starts from depot (node 0) and iteratively adds the
        farthest node from all existing landmarks.

        Args:
            G: The graph
            L: Number of landmarks to select

        Returns:
            List of landmark node indices
        """
        if self.n <= 0 or L <= 0:
            return []

        landmarks: List[int] = [0]  # Start with depot
        used = {0}

        for _ in range(L - 1):
            last = landmarks[-1]
            dist = nx.single_source_dijkstra_path_length(G, last, weight="dist")

            farthest = -1
            best_d = -1.0
            for v, d in dist.items():
                if v not in used and d > best_d:
                    best_d = d
                    farthest = v

            if farthest == -1:
                break

            landmarks.append(farthest)
            used.add(farthest)

        return landmarks

    def a_hat(self, u: int, v: int) -> float:
        """
        Lower bound on shortest path distance d(u, v).

        Uses triangle inequality: d(u,v) >= |d(ell,u) - d(ell,v)|
        We use: a_hat(u,v) = min_ell(A[ell][u] + A[ell][v])

        Args:
            u: Source node
            v: Destination node

        Returns:
            Lower bound on distance
        """
        if u == v:
            return 0.0

        best = float("inf")
        for ell_idx in range(len(self.landmarks)):
            a_u = self._A[ell_idx].get(u, float("inf"))
            a_v = self._A[ell_idx].get(v, float("inf"))
            best = min(best, a_u + a_v)
        return best

    def b_hat(self, u: int, v: int) -> float:
        """
        Lower bound on d^beta metric distance.

        Args:
            u: Source node
            v: Destination node

        Returns:
            Lower bound on d^beta distance
        """
        if u == v:
            return 0.0

        best = float("inf")
        for ell_idx in range(len(self.landmarks)):
            b_u = self._B[ell_idx].get(u, float("inf"))
            b_v = self._B[ell_idx].get(v, float("inf"))
            best = min(best, b_u + b_v)
        return best

    def leg_cost(self, u: int, v: int, w: float) -> float:
        """
        Approximate cost of traversing from u to v with load w.

        Uses: a_hat(u,v) + ((alpha * w)^beta) * b_hat(u,v)

        Args:
            u: Source node
            v: Destination node
            w: Current load

        Returns:
            Approximate leg cost (lower bound)
        """
        a = self.a_hat(u, v)
        b = self.b_hat(u, v)
        return a + ((self.alpha * w) ** self.beta) * b

    def a0(self, i: int) -> float:
        """Get precomputed distance from depot to node i."""
        return self._A0[i] if 0 <= i < self.n else float("inf")

    def b0(self, i: int) -> float:
        """Get precomputed d^beta distance from depot to node i."""
        return self._B0[i] if 0 <= i < self.n else float("inf")

    def _bucket(self, w: float) -> int:
        """Bucket load value for cache efficiency."""
        return int(w / self._bucket_step) * int(self._bucket_step)

    def exact_leg_cost(self, u: int, v: int, w: float) -> float:
        """
        Compute exact cost of traversing from u to v with load w.

        Uses Dijkstra with load-dependent edge weights.
        Results are cached by (u, v, bucketed_w) for efficiency.

        Args:
            u: Source node
            v: Destination node
            w: Current load

        Returns:
            Exact leg cost
        """
        key = (u, v, self._bucket(w))
        if key in self._exact_cache:
            return self._exact_cache[key]

        G = self.problem.graph

        if u == v:
            self._exact_cache[key] = 0.0
            return 0.0

        def weight(x, y, d):
            dist = d.get("dist", 1.0)
            return dist + (self.alpha * w * dist) ** self.beta

        try:
            cost = nx.dijkstra_path_length(G, u, v, weight=weight)
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            cost = float("inf")

        self._exact_cache[key] = cost
        return cost

    def clear_cache(self) -> None:
        """Clear the exact cost cache."""
        self._exact_cache.clear()

    def __repr__(self) -> str:
        return (
            f"LandmarkOracle(n={self.n}, L={len(self.landmarks)}, "
            f"mode={self.plan.mode})"
        )


def build_oracle(
    problem: Problem,
    plan: Optional[OraclePlan] = None,
    L: Optional[int] = None,
) -> LandmarkOracle:
    """
    Build a landmark oracle for the given problem.

    Args:
        problem: The problem instance
        plan: Oracle configuration (computed if not provided)
        L: Number of landmarks (overrides plan if provided)

    Returns:
        Configured LandmarkOracle instance
    """
    if plan is None:
        plan = profile(problem)
    return LandmarkOracle(problem, problem.beta, L=L, plan=plan)