"""Genetic algorithm based solver for the Gold Collection Problem."""

from .operators import (
    tournament_selection,
    order_crossover,
    swap_mutation,
    inversion_mutation,
    insert_mutation,
    mutate,
)
from .split import optimal_split, build_distance_matrix, compute_split_limit
from .tour_solver import solve_with_genetic_algorithm
from .refinement import refine_tour, iterated_local_search

__all__ = [
    "tournament_selection",
    "order_crossover",
    "swap_mutation",
    "inversion_mutation",
    "insert_mutation",
    "mutate",
    "optimal_split",
    "build_distance_matrix",
    "compute_split_limit",
    "solve_with_genetic_algorithm",
    "refine_tour",
    "iterated_local_search",
]