"""
Benchmarking framework for the Gold Collection Problem.

Provides functions to run experiments and collect metrics.
"""

import os
import sys
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

sys.path.insert(0, str(__file__).rsplit("/", 2)[0] + "/src")

from gold_collection import Problem, solve
from gold_collection.solvers.genetic import solve_with_genetic_algorithm
from .instance_generator import build_problem
from .configs import get_instance_configs, InstanceConfig


@dataclass
class BenchmarkResult:
    """
    Result of a single benchmark run.

    Attributes:
        config: Instance configuration
        seed: Random seed used
        baseline_cost: Cost of naive baseline solution
        strategy_cost: Cost of optimized solution
        genetic_cost: Cost of GA-based solution (if run)
        baseline_time: Time for baseline computation
        strategy_time: Time for strategy computation
        improvement_pct: Percentage improvement over baseline
        improvement_ratio: Ratio of baseline to strategy cost
    """

    config: Tuple[int, float, float, float]
    seed: int
    baseline_cost: float
    strategy_cost: float
    genetic_cost: Optional[float] = None
    baseline_time: float = 0.0
    strategy_time: float = 0.0
    genetic_time: float = 0.0

    @property
    def improvement_pct(self) -> float:
        """Percentage improvement over baseline."""
        if self.baseline_cost <= 0:
            return 0.0
        return 100.0 * (self.baseline_cost - self.strategy_cost) / self.baseline_cost

    @property
    def improvement_ratio(self) -> float:
        """Ratio of baseline to strategy cost."""
        if self.strategy_cost <= 0:
            return float("inf")
        return self.baseline_cost / self.strategy_cost


@dataclass
class AggregatedResult:
    """
    Aggregated results for a configuration across multiple seeds.

    Attributes:
        config: Instance configuration
        n_seeds: Number of seeds tested
        baseline_mean: Mean baseline cost
        baseline_std: Std dev of baseline cost
        strategy_mean: Mean strategy cost
        strategy_std: Std dev of strategy cost
        improvement_pct_mean: Mean improvement percentage
        improvement_ratio_mean: Mean improvement ratio
        baseline_time_mean: Mean baseline time
        strategy_time_mean: Mean strategy time
    """

    config: Tuple[int, float, float, float]
    n_seeds: int
    baseline_mean: float
    baseline_std: float
    strategy_mean: float
    strategy_std: float
    improvement_pct_mean: float
    improvement_ratio_mean: float
    baseline_time_mean: float
    strategy_time_mean: float
    genetic_mean: Optional[float] = None
    genetic_std: Optional[float] = None


def run_single_instance(
    num_cities: int,
    density: float,
    alpha: float,
    beta: float,
    seed: int,
    time_budget_s: float = 120.0,
    run_genetic: bool = False,
) -> BenchmarkResult:
    """
    Run benchmark on a single instance.

    Args:
        num_cities: Number of nodes
        density: Edge probability
        alpha: Load penalty scale
        beta: Load penalty exponent
        seed: Random seed
        time_budget_s: Time budget for solver
        run_genetic: Also run GA-based solver

    Returns:
        BenchmarkResult with costs and times
    """
    problem = build_problem(num_cities, density, alpha, beta, seed)

    # Baseline
    t0 = time.perf_counter()
    baseline_cost = problem.baseline()
    t1 = time.perf_counter()

    # Strategy
    strategy_cost = solve(problem, time_budget_s=time_budget_s)
    t2 = time.perf_counter()

    # Optional GA solver
    genetic_cost = None
    genetic_time = 0.0
    if run_genetic:
        t3 = time.perf_counter()
        genetic_cost = solve_with_genetic_algorithm(problem, rng_seed=seed)
        genetic_time = time.perf_counter() - t3

    return BenchmarkResult(
        config=(num_cities, density, alpha, beta),
        seed=seed,
        baseline_cost=baseline_cost,
        strategy_cost=strategy_cost,
        genetic_cost=genetic_cost,
        baseline_time=t1 - t0,
        strategy_time=t2 - t1,
        genetic_time=genetic_time,
    )


def run_configuration(
    config: InstanceConfig,
    n_seeds: int = 10,
    time_budget_s: float = 120.0,
    run_genetic: bool = False,
) -> AggregatedResult:
    """
    Run benchmark on a configuration across multiple seeds.

    Args:
        config: Instance configuration
        n_seeds: Number of random seeds
        time_budget_s: Time budget per run
        run_genetic: Also run GA-based solver

    Returns:
        AggregatedResult with statistics
    """
    num_cities, density, alpha, beta = config
    results: List[BenchmarkResult] = []

    for seed in range(n_seeds):
        result = run_single_instance(
            num_cities, density, alpha, beta, seed,
            time_budget_s=time_budget_s,
            run_genetic=run_genetic,
        )
        results.append(result)

    baseline_costs = np.array([r.baseline_cost for r in results])
    strategy_costs = np.array([r.strategy_cost for r in results])
    baseline_times = np.array([r.baseline_time for r in results])
    strategy_times = np.array([r.strategy_time for r in results])

    genetic_mean = None
    genetic_std = None
    if run_genetic:
        genetic_costs = [r.genetic_cost for r in results if r.genetic_cost is not None]
        if genetic_costs:
            genetic_mean = float(np.mean(genetic_costs))
            genetic_std = float(np.std(genetic_costs))

    return AggregatedResult(
        config=(num_cities, density, alpha, beta),
        n_seeds=n_seeds,
        baseline_mean=float(np.mean(baseline_costs)),
        baseline_std=float(np.std(baseline_costs)),
        strategy_mean=float(np.mean(strategy_costs)),
        strategy_std=float(np.std(strategy_costs)),
        improvement_pct_mean=float(
            100.0 * (np.mean(baseline_costs) - np.mean(strategy_costs)) / np.mean(baseline_costs)
        ) if np.mean(baseline_costs) > 0 else 0.0,
        improvement_ratio_mean=float(np.mean(baseline_costs) / np.mean(strategy_costs))
        if np.mean(strategy_costs) > 0 else float("inf"),
        baseline_time_mean=float(np.mean(baseline_times)),
        strategy_time_mean=float(np.mean(strategy_times)),
        genetic_mean=genetic_mean,
        genetic_std=genetic_std,
    )


def run_full_benchmark(
    n_seeds: int = 10,
    include_hard: bool = False,
    time_budget_s: float = 120.0,
    run_genetic: bool = False,
) -> List[AggregatedResult]:
    """
    Run full benchmark across all configurations.

    Args:
        n_seeds: Number of seeds per configuration
        include_hard: Include hard configurations
        time_budget_s: Time budget per run
        run_genetic: Also run GA-based solver

    Returns:
        List of AggregatedResult for each configuration
    """
    from .configs import BASE_CONFIGS, HARD_CONFIGS

    configs = list(BASE_CONFIGS) + (list(HARD_CONFIGS) if include_hard else [])
    results: List[AggregatedResult] = []

    for config in configs:
        result = run_configuration(
            config, n_seeds=n_seeds,
            time_budget_s=time_budget_s,
            run_genetic=run_genetic,
        )
        results.append(result)

    return results


def print_results(results: List[AggregatedResult]) -> None:
    """
    Print benchmark results in a formatted table.

    Args:
        results: List of aggregated results
    """
    print("\n" + "=" * 100)
    print("BENCHMARK RESULTS")
    print("=" * 100)

    for r in results:
        n, d, a, b = r.config
        line = (
            f"n={n:4d} d={d:.1f} α={a:.1f} β={b:.1f}: "
            f"baseline {r.baseline_mean:10.0f}±{r.baseline_std:6.0f}  "
            f"strategy {r.strategy_mean:10.0f}±{r.strategy_std:6.0f}  "
            f"ratio={r.improvement_ratio_mean:5.2f}  "
            f"improve={r.improvement_pct_mean:5.1f}%  "
            f"t_base={r.baseline_time_mean:.2f}s  "
            f"t_str={r.strategy_time_mean:.1f}s"
        )
        if r.genetic_mean is not None:
            line += f"  genetic={r.genetic_mean:.0f}"
        print(line)

    print("=" * 100)