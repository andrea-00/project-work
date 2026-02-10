#!/usr/bin/env python3
"""
Compare adaptive solver with genetic algorithm solver.

Usage:
    python -m experiments.run_genetic_comparison
    python -m experiments.run_genetic_comparison --cities 50 --seeds 5
"""

import sys
import os
import argparse
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
from gold_collection import Problem, solve
from gold_collection.solvers.genetic import solve_with_genetic_algorithm


def run_comparison(n_cities: int, n_seeds: int, time_budget: float):
    """Compare adaptive and genetic solvers."""

    beta_values = [0.5, 1.0, 2.0]
    results = []

    for beta in beta_values:
        print(f"\n{'='*60}")
        print(f"Testing β = {beta}")
        print('='*60)

        adaptive_costs = []
        genetic_costs = []
        adaptive_times = []
        genetic_times = []

        for seed in range(n_seeds):
            problem = Problem(n_cities, alpha=1.0, beta=beta, density=0.5, seed=seed)
            baseline = problem.baseline()

            # Adaptive solver
            t0 = time.perf_counter()
            adaptive_cost = solve(problem, time_budget_s=time_budget, rng_seed=seed)
            adaptive_time = time.perf_counter() - t0
            adaptive_costs.append(adaptive_cost)
            adaptive_times.append(adaptive_time)

            # Genetic solver
            t0 = time.perf_counter()
            genetic_cost = solve_with_genetic_algorithm(problem, rng_seed=seed)
            genetic_time = time.perf_counter() - t0
            genetic_costs.append(genetic_cost)
            genetic_times.append(genetic_time)

            adaptive_imp = 100 * (baseline - adaptive_cost) / baseline
            genetic_imp = 100 * (baseline - genetic_cost) / baseline

            print(f"  Seed {seed}: Adaptive {adaptive_imp:+.1f}%, Genetic {genetic_imp:+.1f}%")

        results.append({
            'beta': beta,
            'adaptive_mean': np.mean(adaptive_costs),
            'adaptive_std': np.std(adaptive_costs),
            'genetic_mean': np.mean(genetic_costs),
            'genetic_std': np.std(genetic_costs),
            'adaptive_time': np.mean(adaptive_times),
            'genetic_time': np.mean(genetic_times),
            'baseline': baseline,
        })

    return results


def print_results(results):
    """Print comparison table."""
    print("\n" + "=" * 80)
    print("SOLVER COMPARISON RESULTS")
    print("=" * 80)
    print(f"{'Beta':<6} {'Adaptive Cost':<18} {'Genetic Cost':<18} {'Winner':<10} {'Δ%':<8}")
    print("-" * 80)

    for r in results:
        beta = r['beta']
        adapt = f"{r['adaptive_mean']:.0f} ± {r['adaptive_std']:.0f}"
        genet = f"{r['genetic_mean']:.0f} ± {r['genetic_std']:.0f}"

        if r['adaptive_mean'] < r['genetic_mean']:
            winner = "Adaptive"
            delta = 100 * (r['genetic_mean'] - r['adaptive_mean']) / r['genetic_mean']
        else:
            winner = "Genetic"
            delta = 100 * (r['adaptive_mean'] - r['genetic_mean']) / r['adaptive_mean']

        print(f"{beta:<6.1f} {adapt:<18} {genet:<18} {winner:<10} {delta:+.1f}%")

    print("=" * 80)

    # Timing comparison
    print("\nTIMING COMPARISON")
    print("-" * 40)
    print(f"{'Beta':<6} {'Adaptive (s)':<15} {'Genetic (s)':<15}")
    print("-" * 40)
    for r in results:
        print(f"{r['beta']:<6.1f} {r['adaptive_time']:<15.1f} {r['genetic_time']:<15.1f}")
    print("-" * 40)


def main():
    parser = argparse.ArgumentParser(
        description="Compare adaptive and genetic algorithm solvers"
    )
    parser.add_argument(
        "--cities", "-n",
        type=int,
        default=30,
        help="Number of cities (default: 30)"
    )
    parser.add_argument(
        "--seeds", "-s",
        type=int,
        default=3,
        help="Number of seeds per test (default: 3)"
    )
    parser.add_argument(
        "--time", "-t",
        type=float,
        default=30.0,
        help="Time budget for adaptive solver (default: 30)"
    )

    args = parser.parse_args()

    print("=" * 80)
    print("GOLD COLLECTION SOLVER - ADAPTIVE vs GENETIC COMPARISON")
    print("=" * 80)
    print(f"\nSettings:")
    print(f"  Cities:      {args.cities}")
    print(f"  Seeds:       {args.seeds}")
    print(f"  Time budget: {args.time}s (adaptive)")

    results = run_comparison(args.cities, args.seeds, args.time)
    print_results(results)


if __name__ == "__main__":
    main()