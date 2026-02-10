#!/usr/bin/env python3
"""
Quick test script to verify the solver works correctly.

Runs a small instance for each regime (T, L, S) and prints results.

Usage:
    python -m experiments.run_quick_test
"""

import sys
import os
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from gold_collection import Problem, solve
from gold_collection.solvers.adaptive_solver import _choose_regime, Regime
from gold_collection.distance import build_oracle


def test_regime(name: str, beta: float, n_cities: int = 20):
    """Test a specific regime."""
    print(f"\n{'='*60}")
    print(f"Testing Regime {name} (β = {beta})")
    print('='*60)

    problem = Problem(n_cities, alpha=1.0, beta=beta, density=0.5, seed=42)

    # Verify regime selection
    oracle = build_oracle(problem)
    regime = _choose_regime(problem, oracle)
    print(f"Selected regime: {regime.value}")

    # Compute baseline
    t0 = time.perf_counter()
    baseline = problem.baseline()
    t_baseline = time.perf_counter() - t0

    # Compute optimized solution
    t0 = time.perf_counter()
    optimized = solve(problem, time_budget_s=30.0)
    t_solve = time.perf_counter() - t0

    # Results
    improvement = 100 * (baseline - optimized) / baseline

    print(f"\nResults:")
    print(f"  Cities:      {n_cities}")
    print(f"  Total gold:  {problem.total_gold:.0f}")
    print(f"  Baseline:    {baseline:.2f} ({t_baseline:.3f}s)")
    print(f"  Optimized:   {optimized:.2f} ({t_solve:.2f}s)")
    print(f"  Improvement: {improvement:.1f}%")

    return improvement


def main():
    print("=" * 60)
    print("GOLD COLLECTION SOLVER - QUICK TEST")
    print("=" * 60)

    results = {}

    # Test each regime
    results['T'] = test_regime('T', beta=0.5)  # Sub-linear
    results['L'] = test_regime('L', beta=1.0)  # Linear
    results['S'] = test_regime('S', beta=2.0)  # Super-linear

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for regime, improvement in results.items():
        status = "✓ PASS" if improvement >= -5 else "✗ FAIL"
        print(f"  Regime {regime}: {improvement:+.1f}% improvement  {status}")

    print("\n" + "=" * 60)
    all_passed = all(imp >= -5 for imp in results.values())
    if all_passed:
        print("All tests PASSED!")
    else:
        print("Some tests FAILED!")
    print("=" * 60)

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())