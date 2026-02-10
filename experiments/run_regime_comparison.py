#!/usr/bin/env python3
"""
Compare solver performance across different beta values.

Tests multiple beta values and visualizes the improvement trend.

Usage:
    python -m experiments.run_regime_comparison
    python -m experiments.run_regime_comparison --cities 100 --seeds 3
"""

import sys
import os
import argparse
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np
from gold_collection import Problem, solve


def run_comparison(n_cities: int, n_seeds: int, time_budget: float):
    """Run comparison across beta values."""

    beta_values = [0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 2.5, 3.0]

    results = []

    for beta in beta_values:
        print(f"\nTesting β = {beta}...")

        improvements = []
        times = []

        for seed in range(n_seeds):
            problem = Problem(n_cities, alpha=1.0, beta=beta, density=0.5, seed=seed)

            baseline = problem.baseline()

            t0 = time.perf_counter()
            optimized = solve(problem, time_budget_s=time_budget, rng_seed=seed)
            t_solve = time.perf_counter() - t0

            improvement = 100 * (baseline - optimized) / baseline
            improvements.append(improvement)
            times.append(t_solve)

            print(f"  Seed {seed}: {improvement:+.1f}%")

        results.append({
            'beta': beta,
            'improvement_mean': np.mean(improvements),
            'improvement_std': np.std(improvements),
            'time_mean': np.mean(times),
        })

    return results


def print_results(results):
    """Print results table."""
    print("\n" + "=" * 70)
    print("REGIME COMPARISON RESULTS")
    print("=" * 70)
    print(f"{'Beta':<8} {'Regime':<8} {'Improvement':<20} {'Time (s)':<10}")
    print("-" * 70)

    for r in results:
        beta = r['beta']
        regime = 'T' if beta < 1 else ('L' if beta == 1 else 'S')
        imp = f"{r['improvement_mean']:+.1f}% ± {r['improvement_std']:.1f}%"
        print(f"{beta:<8.1f} {regime:<8} {imp:<20} {r['time_mean']:<10.1f}")

    print("=" * 70)


def print_ascii_chart(results):
    """Print ASCII bar chart of improvements."""
    print("\n" + "=" * 70)
    print("IMPROVEMENT BY BETA (ASCII CHART)")
    print("=" * 70)

    max_imp = max(r['improvement_mean'] for r in results)
    min_imp = min(r['improvement_mean'] for r in results)

    # Normalize to 0-50 bar length
    def bar_length(imp):
        if max_imp == min_imp:
            return 25
        return int(50 * (imp - min_imp) / (max_imp - min_imp))

    for r in results:
        beta = r['beta']
        imp = r['improvement_mean']
        bar = "█" * bar_length(imp)
        print(f"β={beta:.1f} | {bar} {imp:+.1f}%")

    print("=" * 70)
    print("Legend: T = sub-linear (β<1), L = linear (β=1), S = super-linear (β>1)")


def main():
    parser = argparse.ArgumentParser(
        description="Compare solver across different beta values"
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
        help="Number of seeds per beta (default: 3)"
    )
    parser.add_argument(
        "--time", "-t",
        type=float,
        default=30.0,
        help="Time budget per run in seconds (default: 30)"
    )

    args = parser.parse_args()

    print("=" * 70)
    print("GOLD COLLECTION SOLVER - REGIME COMPARISON")
    print("=" * 70)
    print(f"\nSettings:")
    print(f"  Cities:      {args.cities}")
    print(f"  Seeds:       {args.seeds}")
    print(f"  Time budget: {args.time}s per run")

    results = run_comparison(args.cities, args.seeds, args.time)

    print_results(results)
    print_ascii_chart(results)


if __name__ == "__main__":
    main()