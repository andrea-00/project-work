#!/usr/bin/env python3
"""
Main script to run benchmark evaluation.

Usage:
    python -m experiments.run_evaluation [--seeds N] [--hard] [--time T] [--genetic]

Examples:
    python -m experiments.run_evaluation
    python -m experiments.run_evaluation --seeds 5 --time 60
    python -m experiments.run_evaluation --hard --genetic
"""

import argparse
import os
import sys

from experiments.benchmark import run_full_benchmark, print_results


def main():
    parser = argparse.ArgumentParser(
        description="Run Gold Collection Problem benchmark evaluation"
    )
    parser.add_argument(
        "--seeds", "-n",
        type=int,
        default=5,
        help="Number of random seeds per configuration (default: 5)"
    )
    parser.add_argument(
        "--hard",
        action="store_true",
        help="Include hard (stress test) configurations"
    )
    parser.add_argument(
        "--time", "-t",
        type=float,
        default=120.0,
        help="Time budget per run in seconds (default: 120)"
    )
    parser.add_argument(
        "--genetic", "-g",
        action="store_true",
        help="Also run GA-based solver for comparison"
    )

    args = parser.parse_args()

    print(f"Running benchmark with:")
    print(f"  Seeds per config: {args.seeds}")
    print(f"  Include hard configs: {args.hard}")
    print(f"  Time budget: {args.time}s")
    print(f"  Run genetic solver: {args.genetic}")
    print()

    results = run_full_benchmark(
        n_seeds=args.seeds,
        include_hard=args.hard,
        time_budget_s=args.time,
        run_genetic=args.genetic,
    )

    print_results(results)


if __name__ == "__main__":
    main()