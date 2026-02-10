#!/usr/bin/env python3
"""
Run a single test with custom parameters.

Usage:
    python -m experiments.run_single_test --cities 100 --beta 0.5 --time 60
    python -m experiments.run_single_test -n 50 -b 2.0 -a 1.5 --seed 123
"""

import sys
import os
import argparse
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from gold_collection import Problem, solve
from gold_collection.solvers.adaptive_solver import _choose_regime
from gold_collection.distance import build_oracle


def main():
    parser = argparse.ArgumentParser(
        description="Run a single Gold Collection Problem test"
    )
    parser.add_argument(
        "--cities", "-n",
        type=int,
        default=50,
        help="Number of cities (default: 50)"
    )
    parser.add_argument(
        "--alpha", "-a",
        type=float,
        default=1.0,
        help="Alpha parameter (default: 1.0)"
    )
    parser.add_argument(
        "--beta", "-b",
        type=float,
        default=1.0,
        help="Beta parameter (default: 1.0)"
    )
    parser.add_argument(
        "--density", "-d",
        type=float,
        default=0.5,
        help="Edge density (default: 0.5)"
    )
    parser.add_argument(
        "--seed", "-s",
        type=int,
        default=42,
        help="Random seed (default: 42)"
    )
    parser.add_argument(
        "--time", "-t",
        type=float,
        default=60.0,
        help="Time budget in seconds (default: 60)"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Verbose output"
    )

    args = parser.parse_args()

    print("=" * 60)
    print("GOLD COLLECTION SOLVER - SINGLE TEST")
    print("=" * 60)
    print(f"\nParameters:")
    print(f"  Cities:      {args.cities}")
    print(f"  Alpha:       {args.alpha}")
    print(f"  Beta:        {args.beta}")
    print(f"  Density:     {args.density}")
    print(f"  Seed:        {args.seed}")
    print(f"  Time budget: {args.time}s")

    # Create problem
    print("\nCreating problem instance...")
    problem = Problem(
        args.cities,
        alpha=args.alpha,
        beta=args.beta,
        density=args.density,
        seed=args.seed
    )

    # Show problem info
    print(f"\nProblem info:")
    print(f"  Total gold:  {problem.total_gold:.0f}")
    print(f"  Graph edges: {problem.graph.number_of_edges()}")

    # Determine regime
    oracle = build_oracle(problem)
    regime = _choose_regime(problem, oracle)
    print(f"  Regime:      {regime.value}")

    # Compute baseline
    print("\nComputing baseline...")
    t0 = time.perf_counter()
    baseline = problem.baseline()
    t_baseline = time.perf_counter() - t0
    print(f"  Baseline cost: {baseline:.2f} ({t_baseline:.3f}s)")

    # Compute optimized solution
    print(f"\nOptimizing (time budget: {args.time}s)...")
    t0 = time.perf_counter()
    optimized = solve(problem, time_budget_s=args.time, rng_seed=args.seed)
    t_solve = time.perf_counter() - t0

    # Results
    improvement = 100 * (baseline - optimized) / baseline
    ratio = baseline / optimized if optimized > 0 else float('inf')

    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"  Baseline cost:    {baseline:.2f}")
    print(f"  Optimized cost:   {optimized:.2f}")
    print(f"  Improvement:      {improvement:.2f}%")
    print(f"  Ratio:            {ratio:.3f}x")
    print(f"  Solve time:       {t_solve:.2f}s")
    print("=" * 60)


if __name__ == "__main__":
    main()