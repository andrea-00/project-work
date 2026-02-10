#!/usr/bin/env python3
"""
Run all experiments in sequence.

Usage:
    python -m experiments.run_all_experiments
    python -m experiments.run_all_experiments --quick
"""

import sys
import os
import argparse
import subprocess

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


def run_experiment(name: str, args: list):
    """Run a single experiment."""
    print(f"\n{'#'*70}")
    print(f"# Running: {name}")
    print('#'*70 + "\n")

    cmd = [sys.executable, "-m", f"experiments.{name}"] + args
    result = subprocess.run(cmd, cwd=os.path.dirname(os.path.dirname(__file__)))

    if result.returncode != 0:
        print(f"\n⚠ Experiment {name} finished with return code {result.returncode}")

    return result.returncode


def main():
    parser = argparse.ArgumentParser(
        description="Run all Gold Collection experiments"
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run quick versions of experiments"
    )
    parser.add_argument(
        "--skip-genetic",
        action="store_true",
        help="Skip genetic algorithm comparison"
    )

    args = parser.parse_args()

    print("=" * 70)
    print("GOLD COLLECTION SOLVER - FULL EXPERIMENT SUITE")
    print("=" * 70)

    experiments = []

    # Quick test
    experiments.append(("run_quick_test", []))

    # Regime comparison
    if args.quick:
        experiments.append(("run_regime_comparison", ["--cities", "20", "--seeds", "2", "--time", "10"]))
    else:
        experiments.append(("run_regime_comparison", ["--cities", "30", "--seeds", "3", "--time", "30"]))

    # Scalability test
    if args.quick:
        experiments.append(("run_scalability_test", ["--max-cities", "50", "--time", "10"]))
    else:
        experiments.append(("run_scalability_test", ["--max-cities", "200", "--time", "30"]))

    # Genetic comparison
    if not args.skip_genetic:
        if args.quick:
            experiments.append(("run_genetic_comparison", ["--cities", "20", "--seeds", "2", "--time", "10"]))
        else:
            experiments.append(("run_genetic_comparison", ["--cities", "30", "--seeds", "3", "--time", "30"]))

    # Full benchmark
    if args.quick:
        experiments.append(("run_evaluation", ["--seeds", "2", "--time", "15"]))
    else:
        experiments.append(("run_evaluation", ["--seeds", "3", "--time", "60"]))

    # Run all experiments
    results = {}
    for name, exp_args in experiments:
        results[name] = run_experiment(name, exp_args)

    # Summary
    print("\n" + "=" * 70)
    print("EXPERIMENT SUMMARY")
    print("=" * 70)

    for name, returncode in results.items():
        status = "✓ SUCCESS" if returncode == 0 else f"✗ FAILED ({returncode})"
        print(f"  {name}: {status}")

    print("=" * 70)

    all_passed = all(rc == 0 for rc in results.values())
    if all_passed:
        print("\nAll experiments completed successfully!")
    else:
        print("\nSome experiments failed. Check output above for details.")

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())