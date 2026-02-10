"""Experiment framework for the Gold Collection Problem."""

from .configs import (
    BASE_CONFIGS,
    HARD_CONFIGS,
    get_instance_configs,
    InstanceConfig,
)
from .instance_generator import build_problem
from .benchmark import (
    run_single_instance,
    run_configuration,
    run_full_benchmark,
    BenchmarkResult,
)

__all__ = [
    "BASE_CONFIGS",
    "HARD_CONFIGS",
    "get_instance_configs",
    "InstanceConfig",
    "build_problem",
    "run_single_instance",
    "run_configuration",
    "run_full_benchmark",
    "BenchmarkResult",
]