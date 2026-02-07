from problem_base import Problem
from hierarchical_gold.giant_tour_solver import solve_giant_tour_prins


def solution(problem: Problem) -> float:
    return solve_giant_tour_prins(problem)
