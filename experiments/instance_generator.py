"""
Instance generation for experiments.

Provides functions to create problem instances for benchmarking.
"""

import sys
sys.path.insert(0, str(__file__).rsplit("/", 2)[0] + "/src")

from gold_collection import Problem


def build_problem(
    num_cities: int,
    density: float,
    alpha: float,
    beta: float,
    seed: int,
) -> Problem:
    """
    Build a problem instance from parameters.

    Args:
        num_cities: Number of nodes (including depot)
        density: Edge probability
        alpha: Load penalty scale
        beta: Load penalty exponent
        seed: Random seed

    Returns:
        Configured Problem instance
    """
    return Problem(
        num_cities,
        alpha=alpha,
        beta=beta,
        density=density,
        seed=seed,
    )


def build_small_test_instance(seed: int = 42) -> Problem:
    """
    Build a small test instance for quick validation.

    Args:
        seed: Random seed

    Returns:
        Small Problem instance (10 cities)
    """
    return Problem(10, alpha=1.0, beta=1.0, density=0.5, seed=seed)


def build_medium_test_instance(seed: int = 42) -> Problem:
    """
    Build a medium test instance.

    Args:
        seed: Random seed

    Returns:
        Medium Problem instance (100 cities)
    """
    return Problem(100, alpha=1.0, beta=1.0, density=0.5, seed=seed)


def build_large_test_instance(seed: int = 42) -> Problem:
    """
    Build a large test instance for stress testing.

    Args:
        seed: Random seed

    Returns:
        Large Problem instance (1000 cities)
    """
    return Problem(1000, alpha=1.0, beta=1.0, density=0.3, seed=seed)