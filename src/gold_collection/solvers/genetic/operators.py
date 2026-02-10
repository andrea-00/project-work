"""
Genetic algorithm operators for permutation-based representation.

Provides selection, crossover, and mutation operators for evolving
tour permutations.
"""

import random
from typing import List, Callable, Tuple

Genotype = List[int]


def tournament_selection(
    population: List[Genotype],
    fitnesses: List[float],
    tournament_size: int,
    n_select: int,
) -> List[Genotype]:
    """
    Select individuals using tournament selection.

    Args:
        population: List of genotypes
        fitnesses: Fitness values (higher is better)
        tournament_size: Number of individuals per tournament
        n_select: Number of individuals to select

    Returns:
        Selected individuals
    """
    parents = []
    for _ in range(n_select):
        idxs = random.sample(range(len(population)), min(tournament_size, len(population)))
        best_i = max(idxs, key=lambda i: fitnesses[i])
        parents.append(list(population[best_i]))
    return parents


def order_crossover(p1: Genotype, p2: Genotype) -> Genotype:
    """
    Order crossover (OX) for permutations.

    Preserves relative order of elements not in the copied segment.

    Args:
        p1: First parent
        p2: Second parent

    Returns:
        Offspring genotype
    """
    n = len(p1)
    if n <= 1:
        return list(p1)

    a, b = sorted(random.sample(range(n), 2))
    child = [None] * n
    child[a:b] = p1[a:b]

    remaining = [x for x in p2 if x not in child[a:b]]
    j = 0
    for i in range(n):
        if child[i] is None:
            child[i] = remaining[j]
            j += 1

    return child


def inversion_crossover(p1: Genotype, p2: Genotype) -> Genotype:
    """
    Inversion-based crossover using successor information from p2.

    Args:
        p1: First parent
        p2: Second parent

    Returns:
        Offspring genotype
    """
    n = len(p1)
    if n <= 1:
        return list(p1)

    p2_succ = {p2[i]: p2[(i + 1) % n] for i in range(n)}
    child = list(p1)

    for _ in range(n):
        c1 = random.choice(child)
        c2 = p2_succ.get(c1)
        if c2 is None or c2 not in child:
            break
        i, j = child.index(c1), child.index(c2)
        if i > j:
            i, j = j, i
        child[i : j + 1] = reversed(child[i : j + 1])

    return child


def swap_mutation(genotype: Genotype, prob: float) -> Genotype:
    """
    Swap mutation: exchange two random positions.

    Args:
        genotype: Input genotype
        prob: Probability of applying mutation

    Returns:
        Possibly mutated genotype
    """
    g = list(genotype)
    if random.random() >= prob or len(g) < 2:
        return g

    i, j = random.sample(range(len(g)), 2)
    g[i], g[j] = g[j], g[i]
    return g


def inversion_mutation(genotype: Genotype, prob: float) -> Genotype:
    """
    Inversion (2-opt) mutation: reverse a random segment.

    Args:
        genotype: Input genotype
        prob: Probability of applying mutation

    Returns:
        Possibly mutated genotype
    """
    g = list(genotype)
    if random.random() >= prob or len(g) < 2:
        return g

    i, j = sorted(random.sample(range(len(g)), 2))
    g[i : j + 1] = reversed(g[i : j + 1])
    return g


def insert_mutation(genotype: Genotype, prob: float) -> Genotype:
    """
    Insertion mutation: move an element to a new position.

    Args:
        genotype: Input genotype
        prob: Probability of applying mutation

    Returns:
        Possibly mutated genotype
    """
    g = list(genotype)
    if random.random() >= prob or len(g) < 2:
        return g

    i = random.randrange(len(g))
    j = random.randrange(len(g))
    if i == j:
        return g

    x = g.pop(i)
    g.insert(j, x)
    return g


def mutate(
    genotype: Genotype,
    swap_prob: float = 0.2,
    inversion_prob: float = 0.2,
    insert_prob: float = 0.0,
) -> Genotype:
    """
    Apply multiple mutation operators.

    Args:
        genotype: Input genotype
        swap_prob: Probability of swap mutation
        inversion_prob: Probability of inversion mutation
        insert_prob: Probability of insertion mutation

    Returns:
        Mutated genotype
    """
    g = swap_mutation(genotype, swap_prob)
    g = inversion_mutation(g, inversion_prob)
    g = insert_mutation(g, insert_prob)
    return g


def run_genetic_algorithm(
    fitness_fn: Callable[[Genotype], float],
    initializer_fn: Callable[[], Genotype],
    n_generations: int,
    population_size: int,
    *,
    tournament_size: int = 3,
    crossover_prob: float = 0.8,
    swap_mut_prob: float = 0.2,
    inversion_mut_prob: float = 0.2,
    rng_seed: int | None = None,
) -> Tuple[Genotype, float]:
    """
    Run a genetic algorithm to optimize a permutation.

    Args:
        fitness_fn: Function to evaluate fitness (higher is better)
        initializer_fn: Function to create initial genotypes
        n_generations: Number of generations
        population_size: Population size
        tournament_size: Tournament size for selection
        crossover_prob: Crossover probability
        swap_mut_prob: Swap mutation probability
        inversion_mut_prob: Inversion mutation probability
        rng_seed: Random seed

    Returns:
        Tuple of (best genotype, best fitness)
    """
    if rng_seed is not None:
        random.seed(rng_seed)

    # Initialize population
    pop = [initializer_fn() for _ in range(population_size)]
    fitnesses = [fitness_fn(g) for g in pop]

    best_i = max(range(len(pop)), key=lambda i: fitnesses[i])
    best_genotype = list(pop[best_i])
    best_fitness = fitnesses[best_i]

    for gen in range(n_generations - 1):
        # Selection
        parents = tournament_selection(pop, fitnesses, tournament_size, population_size)

        # Crossover and mutation
        offspring = []
        for i in range(0, population_size, 2):
            p1, p2 = parents[i], parents[i + 1] if i + 1 < population_size else parents[0]

            if random.random() < crossover_prob:
                c1 = order_crossover(p1, p2)
                c2 = order_crossover(p2, p1)
            else:
                c1, c2 = list(p1), list(p2)

            c1 = mutate(c1, swap_mut_prob, inversion_mut_prob, 0.0)
            c2 = mutate(c2, swap_mut_prob, inversion_mut_prob, 0.0)

            offspring.append(c1)
            if len(offspring) < population_size:
                offspring.append(c2)

        offspring = offspring[:population_size]
        off_fitnesses = [fitness_fn(g) for g in offspring]

        # Elitist selection
        combined = list(zip(pop + offspring, fitnesses + off_fitnesses))
        combined.sort(key=lambda x: x[1], reverse=True)
        pop = [list(x[0]) for x in combined[:population_size]]
        fitnesses = [x[1] for x in combined[:population_size]]

        if fitnesses[0] > best_fitness:
            best_fitness = fitnesses[0]
            best_genotype = list(pop[0])

    return best_genotype, best_fitness
