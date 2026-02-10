"""
Gold Collection Problem Solver

A computational intelligence approach to solve the Gold Collection Problem,
a variant of the Vehicle Routing Problem with load-dependent edge costs.

Authors: Alberto Migliorato, Andrea Di Felice
Course: Computational Intelligence 2026
"""

from .core.problem import Problem
from .core.solution import Trip, Solution
from .core.evaluator import Evaluator
from .solvers.adaptive_solver import solve, get_solution_cost
from .distance.oracle import LandmarkOracle, build_oracle

__version__ = "1.0.0"

__all__ = [
    "Problem",
    "Trip",
    "Solution",
    "Evaluator",
    "solve",
    "get_solution_cost",
    "LandmarkOracle",
    "build_oracle",
]