"""
Instance configurations for experiments.

Defines standard configurations for benchmarking the solver.
"""

from dataclasses import dataclass
from typing import Iterator, List, Tuple


@dataclass(frozen=True)
class InstanceConfig:
    """
    Configuration for a problem instance.

    Attributes:
        num_cities: Number of nodes (including depot)
        density: Edge probability
        alpha: Load penalty scale
        beta: Load penalty exponent
    """

    num_cities: int
    density: float
    alpha: float
    beta: float

    def __iter__(self):
        """Allow unpacking as tuple."""
        return iter((self.num_cities, self.density, self.alpha, self.beta))


# Base configurations for standard testing
BASE_CONFIGS: List[InstanceConfig] = [
    # Small instances (n=100)
    InstanceConfig(100, 0.2, 1.0, 0.5),
    InstanceConfig(100, 0.2, 1.0, 1.0),
    InstanceConfig(100, 0.2, 1.0, 2.0),
    InstanceConfig(100, 0.2, 2.0, 1.0),
    InstanceConfig(100, 0.2, 2.0, 2.0),
    InstanceConfig(100, 1.0, 1.0, 0.5),
    InstanceConfig(100, 1.0, 1.0, 1.0),
    InstanceConfig(100, 1.0, 1.0, 2.0),
    InstanceConfig(100, 1.0, 2.0, 1.0),
    InstanceConfig(100, 1.0, 2.0, 2.0),
    # Medium instances (n=1000)
    InstanceConfig(1000, 0.2, 1.0, 0.5),
    InstanceConfig(1000, 0.2, 1.0, 1.0),
    InstanceConfig(1000, 0.2, 1.0, 2.0),
    InstanceConfig(1000, 1.0, 1.0, 1.0),
    InstanceConfig(1000, 1.0, 2.0, 1.0),
    InstanceConfig(1000, 1.0, 1.0, 2.0),
]

# Hard configurations for stress testing
HARD_CONFIGS: List[InstanceConfig] = [
    InstanceConfig(2000, 0.2, 1.0, 1.0),
    InstanceConfig(2000, 0.5, 1.0, 1.0),
    InstanceConfig(2000, 0.2, 2.0, 1.0),
    InstanceConfig(5000, 0.2, 1.0, 1.0),
]


def get_instance_configs(
    n_seeds: int = 10, include_hard: bool = False
) -> Iterator[Tuple[int, float, float, float, int]]:
    """
    Generate instance configurations with seeds.

    Args:
        n_seeds: Number of random seeds per configuration
        include_hard: Include hard (stress test) configurations

    Yields:
        Tuples of (num_cities, density, alpha, beta, seed)
    """
    configs = list(BASE_CONFIGS) + (list(HARD_CONFIGS) if include_hard else [])

    for config in configs:
        for seed in range(n_seeds):
            yield (config.num_cities, config.density, config.alpha, config.beta, seed)


def get_configs_by_regime(regime: str) -> List[InstanceConfig]:
    """
    Get configurations for a specific regime.

    Args:
        regime: 'T' (beta < 1), 'L' (beta = 1), or 'S' (beta > 1)

    Returns:
        List of matching configurations
    """
    if regime == "T":
        return [c for c in BASE_CONFIGS if c.beta < 1]
    elif regime == "L":
        return [c for c in BASE_CONFIGS if c.beta == 1]
    elif regime == "S":
        return [c for c in BASE_CONFIGS if c.beta > 1]
    else:
        raise ValueError(f"Unknown regime: {regime}")