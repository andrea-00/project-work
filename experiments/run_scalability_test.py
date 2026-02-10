#!/usr/bin/env python3
"""
Test solver scalability with increasing problem sizes.

Usage:
    python -m experiments.run_scalability_test
    python -m experiments.run_scalability_test --max-cities 500 --beta 1.0
"""

import sys
import os
import argparse
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from gold_collection import Problem, solve


def run_scalability_test(city_sizes: list, beta: float, time_budget: float, seed: int):
    """Run scalability test across different problem sizes."""

    results = []

    for n_cities in city_sizes:
        print(f"\nTesting n = {n_cities}...")

        # Create problem
        t0 = time.perf_counter()
        problem = Problem(n_cities, alpha=1.0, beta=beta, density=0.5, seed=seed)
        t_create = time.perf_counter() - t0

        # Baseline
        t0 = time.perf_counter()
        baseline = problem.baseline()
        t_baseline = time.perf_counter() - t0

        # Solve
        t0 = time.perf_counter()
        optimized = solve(problem, time_budget_s=time_budget, rng_seed=seed)
        t_solve = time.perf_counter() - t0

        improvement = 100 * (baseline - optimized) / baseline

        results.append({
            'n': n_cities,
            'baseline': baseline,
            'optimized': optimized,
            'improvement': improvement,
            't_create': t_create,
            't_baseline': t_baseline,
            't_solve': t_solve,
        })

        print(f"  Improvement: {improvement:+.1f}%, Solve time: {t_solve:.1f}s")

    return results


def print_results(results, beta):
    """Print results table."""
    print("\n" + "=" * 80)
    print(f"SCALABILITY TEST RESULTS (β = {beta})")
    print("=" * 80)
    print(f"{'Cities':<10} {'Baseline':<15} {'Optimized':<15} {'Improve':<12} {'Time (s)':<10}")
    print("-" * 80)

    for r in results:
        print(f"{r['n']:<10} {r['baseline']:<15.0f} {r['optimized']:<15.0f} "
              f"{r['improvement']:+.1f}%{'':<6} {r['t_solve']:<10.1f}")

    print("=" * 80)


def print_timing_chart(results):
    """Print ASCII chart of solve times."""
    print("\n" + "=" * 80)
    print("SOLVE TIME BY PROBLEM SIZE")
    print("=" * 80)

    max_time = max(r['t_solve'] for r in results)

    for r in results:
        n = r['n']
        t = r['t_solve']
        bar_len = int(50 * t / max_time) if max_time > 0 else 0
        bar = "█" * bar_len
        print(f"n={n:<4} | {bar} {t:.1f}s")

    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(
        description="Test solver scalability with increasing problem sizes"
    )
    parser.add_argument(
        "--max-cities", "-m",
        type=int,
        default=200,
        help="Maximum number of cities (default: 200)"
    )
    parser.add_argument(
        "--beta", "-b",
        type=float,
        default=1.0,
        help="Beta parameter (default: 1.0)"
    )
    parser.add_argument(
        "--time", "-t",
        type=float,
        default=30.0,
        help="Time budget per run in seconds (default: 30)"
    )
    parser.add_argument(
        "--seed", "-s",
        type=int,
        default=42,
        help="Random seed (default: 42)"
    )

    args = parser.parse_args()

    # Generate city sizes
    city_sizes = [10, 20, 50, 100]
    if args.max_cities >= 200:
        city_sizes.append(200)
    if args.max_cities >= 500:
        city_sizes.append(500)
    if args.max_cities >= 1000:
        city_sizes.append(1000)

    city_sizes = [n for n in city_sizes if n <= args.max_cities]

    print("=" * 80)
    print("GOLD COLLECTION SOLVER - SCALABILITY TEST")
    print("=" * 80)
    print(f"\nSettings:")
    print(f"  City sizes:  {city_sizes}")
    print(f"  Beta:        {args.beta}")
    print(f"  Time budget: {args.time}s per run")
    print(f"  Seed:        {args.seed}")

    results = run_scalability_test(city_sizes, args.beta, args.time, args.seed)

    print_results(results, args.beta)
    print_timing_chart(results)


if __name__ == "__main__":
    main()