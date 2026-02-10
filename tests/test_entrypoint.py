"""Test the submission entrypoint s343585.solution()."""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from s343585 import solution
from src.gold_collection.core.problem import Problem


def test_solution_returns_list_of_tuples():
    """solution(problem) returns list of (node_id, gold_amount) tuples."""
    p = Problem(10, alpha=1.0, beta=1.0, density=0.5, seed=42)
    path = solution(p)
    assert isinstance(path, list)
    assert len(path) > 0
    for t in path:
        assert isinstance(t, tuple)
        assert len(t) == 2
        node, gold = t
        assert isinstance(node, int)
        assert isinstance(gold, (int, float))


def test_solution_collects_all_gold():
    """Solution covers all gold (total pickup equals problem total_gold)."""
    p = Problem(15, alpha=1.0, beta=1.0, density=0.5, seed=42)
    path = solution(p)
    total = sum(g for _, g in path)
    assert abs(total - p.total_gold) < 1e-6


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
