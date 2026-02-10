"""
Genetic algorithm based tour solver.

Uses GA to evolve a giant tour, then applies Prins split to partition
into optimal trips.
"""

import random
from typing import List, Tuple, Callable

import numpy as np

from ...core.problem import Problem
from .operators import run_genetic_algorithm, mutate, order_crossover, tournament_selection
from .split import build_distance_matrix, compute_split_limit, optimal_split
from .refinement import refine_tour, iterated_local_search


def _make_fitness_function(
    dist_matrix: np.ndarray,
    golds: np.ndarray,
    alpha: float,
    beta: float,
    limit: int,
) -> Callable[[List[int]], float]:
    """
    Create fitness function for GA.

    Fitness is negative of split cost (higher is better).

    Args:
        dist_matrix: Distance matrix
        golds: Gold quantities
        alpha: Load penalty scale
        beta: Load penalty exponent
        limit: Split window limit

    Returns:
        Fitness function
    """
    def fitness(tour: List[int]) -> float:
        cost_val, _ = optimal_split(tour, dist_matrix, golds, alpha, beta, limit)
        return -cost_val

    return fitness


def _greedy_nearest_neighbor_tour(
    nodes: List[int], dist_matrix: np.ndarray, start_from: int = 0
) -> List[int]:
    """
    Build initial tour using nearest neighbor heuristic.

    Args:
        nodes: List of city indices (no depot)
        dist_matrix: Distance matrix
        start_from: Starting node for NN

    Returns:
        Tour as list of city indices
    """
    unvisited = set(nodes)
    if not unvisited:
        return []

    tour: List[int] = []
    current = start_from

    while unvisited:
        next_node = min(unvisited, key=lambda v: dist_matrix[current, v])
        tour.append(next_node)
        unvisited.discard(next_node)
        current = next_node

    return tour


def run_ga_with_split(
    problem: Problem,
    golds_override: np.ndarray | None = None,
    *,
    population_size_percent: float = 0.23,
    generations_percent: float = 0.21,
    tournament_size_percent: float = 0.14,
    crossover_prob: float = 0.8,
    mutation_swap: float = 0.10,
    mutation_inversion: float = 0.30,
    mutation_insert: float = 0.15,
    warm_start: bool = True,
    rng_seed: int | None = None,
) -> Tuple[List[int], float]:
    """
    Run GA to find best tour, evaluated using Prins split.

    Args:
        problem: The problem instance
        golds_override: Override gold values
        population_size_percent: Population size as fraction of n
        generations_percent: Generations as fraction of n
        tournament_size_percent: Tournament size as fraction of pop
        crossover_prob: Crossover probability
        mutation_swap: Swap mutation probability
        mutation_inversion: Inversion mutation probability
        mutation_insert: Insertion mutation probability
        warm_start: Initialize one individual with NN tour
        rng_seed: Random seed

    Returns:
        Tuple of (best tour, split cost)
    """
    dist_matrix, golds_from_p = build_distance_matrix(problem)
    golds = golds_override if golds_override is not None else golds_from_p

    n = problem.graph.number_of_nodes()
    nodes = list(range(1, n))

    if not nodes:
        return [], 0.0

    limit = compute_split_limit(dist_matrix, golds, problem.alpha, problem.beta, n)
    fitness_fn = _make_fitness_function(
        dist_matrix, golds, problem.alpha, problem.beta, limit
    )

    def initializer() -> List[int]:
        perm = list(nodes)
        random.shuffle(perm)
        return perm

    # Compute sizes based on problem
    p_size = max(2, int(len(nodes) * population_size_percent))
    gens = max(1, int(len(nodes) * generations_percent))
    tournament_size = max(2, int(tournament_size_percent * p_size))

    # Initialize population
    pop = [initializer() for _ in range(p_size)]
    if warm_start and p_size > 0:
        pop[0] = _greedy_nearest_neighbor_tour(nodes, dist_matrix)

    fitnesses = [fitness_fn(g) for g in pop]
    best_i = max(range(len(pop)), key=lambda i: fitnesses[i])
    best_tour = list(pop[best_i])
    best_fitness = fitnesses[best_i]

    # Evolution loop
    for gen in range(gens - 1):
        parents = tournament_selection(pop, fitnesses, tournament_size, p_size)

        offspring = []
        for i in range(0, p_size, 2):
            p1, p2 = parents[i], parents[i + 1] if i + 1 < p_size else parents[0]

            if random.random() < crossover_prob:
                c1 = order_crossover(p1, p2)
                c2 = order_crossover(p2, p1)
            else:
                c1, c2 = list(p1), list(p2)

            c1 = mutate(c1, mutation_swap, mutation_inversion, mutation_insert)
            c2 = mutate(c2, mutation_swap, mutation_inversion, mutation_insert)

            offspring.append(c1)
            if len(offspring) < p_size:
                offspring.append(c2)

        offspring = offspring[:p_size]
        off_fitnesses = [fitness_fn(g) for g in offspring]

        combined = list(zip(pop + offspring, fitnesses + off_fitnesses))
        combined.sort(key=lambda x: x[1], reverse=True)
        pop = [list(x[0]) for x in combined[:p_size]]
        fitnesses = [x[1] for x in combined[:p_size]]

        if fitnesses[0] > best_fitness:
            best_fitness = fitnesses[0]
            best_tour = list(pop[0])

    cost_val, _ = optimal_split(
        best_tour, dist_matrix, golds, problem.alpha, problem.beta, limit
    )

    return best_tour, cost_val


def solve_with_genetic_algorithm(
    problem: Problem,
    *,
    rng_seed: int | None = None,
    multi_start: int = 2,
    use_refinement: bool = True,
    use_ils: bool = True,
    refinement_method: str = "hc",
    refinement_max_evals: int = 500,
    refinement_stall: int = 80,
    ils_restarts: int = 3,
    ils_refine_evals: int = 150,
) -> float:
    """
    Solve the problem using GA + Prins split + refinement.

    Args:
        problem: The problem instance
        rng_seed: Random seed
        multi_start: Number of independent runs
        use_refinement: Apply local refinement
        use_ils: Apply iterated local search
        refinement_method: Refinement method ('hc', 'sa', 'tabu')
        refinement_max_evals: Max evaluations for refinement
        refinement_stall: Stall limit for refinement
        ils_restarts: Number of ILS restarts
        ils_refine_evals: Evaluations per ILS restart

    Returns:
        Best solution cost
    """
    base_seed = rng_seed if rng_seed is not None else 42
    best_cost = float("inf")

    ga_kw = {
        "mutation_swap": 0.10,
        "mutation_inversion": 0.30,
        "mutation_insert": 0.15,
    }

    for run in range(multi_start):
        best_tour, cost_remainder = run_ga_with_split(
            problem, golds_override=None, rng_seed=base_seed + run * 1000, **ga_kw
        )

        total = cost_remainder

        if not best_tour:
            best_cost = min(best_cost, total)
            continue

        rng = random.Random(base_seed + run * 1000)

        if use_refinement:
            best_tour, refined_cost = refine_tour(
                problem,
                best_tour,
                golds_override=None,
                method=refinement_method,
                max_evals=refinement_max_evals,
                stall=refinement_stall,
                accept_equal=True,
                rng=rng,
            )
            total = refined_cost

        if use_ils and ils_restarts > 1:
            best_tour, ils_cost = iterated_local_search(
                problem,
                best_tour,
                total,
                golds_override=None,
                num_restarts=ils_restarts,
                refine_max_evals=refinement_max_evals,
                refine_stall=refinement_stall,
                perturb_strength="medium",
                refine_after_perturb_evals=ils_refine_evals,
                rng=rng,
            )
            total = ils_cost

        best_cost = min(best_cost, total)

    return best_cost