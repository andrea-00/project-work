import random
from typing import List, Callable, Tuple

Genotype = List[int]


def tournament_selection(
    population: List[Genotype],
    fitnesses: List[float],
    tournament_size: int,
    n_select: int,
) -> List[Genotype]:
    parents = []
    for _ in range(n_select):
        idxs = random.sample(range(len(population)), min(tournament_size, len(population)))
        best_i = max(idxs, key=lambda i: fitnesses[i])
        parents.append(list(population[best_i]))
    return parents


def order_crossover(p1: Genotype, p2: Genotype) -> Genotype:
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


def inver_over_crossover(p1: Genotype, p2: Genotype) -> Genotype:
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
    g = list(genotype)
    if random.random() >= prob or len(g) < 2:
        return g
    i, j = random.sample(range(len(g)), 2)
    g[i], g[j] = g[j], g[i]
    return g


def inversion_2opt_mutation(genotype: Genotype, prob: float) -> Genotype:
    g = list(genotype)
    if random.random() >= prob or len(g) < 2:
        return g
    i, j = sorted(random.sample(range(len(g)), 2))
    g[i : j + 1] = reversed(g[i : j + 1])
    return g


def insert_mutation(genotype: Genotype, prob: float) -> Genotype:
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
    two_opt_prob: float = 0.2,
    insert_prob: float = 0.0,
) -> Genotype:
    g = swap_mutation(genotype, swap_prob)
    g = inversion_2opt_mutation(g, two_opt_prob)
    g = insert_mutation(g, insert_prob)
    return g


def run_ga(
    fitness_fn: Callable[[Genotype], float],
    initializer_fn: Callable[[], Genotype],
    n_generations: int,
    population_size: int,
    *,
    tournament_size: int = 3,
    crossover_prob: float = 0.8,
    swap_mut_prob: float = 0.2,
    two_opt_mut_prob: float = 0.2,
    rng_seed: int | None = None,
) -> Tuple[Genotype, float]:
    if rng_seed is not None:
        random.seed(rng_seed)
    pop = [initializer_fn() for _ in range(population_size)]
    fitnesses = [fitness_fn(g) for g in pop]
    best_i = max(range(len(pop)), key=lambda i: fitnesses[i])
    best_genotype = list(pop[best_i])
    best_fitness = fitnesses[best_i]
    for gen in range(n_generations - 1):
        parents = tournament_selection(pop, fitnesses, tournament_size, population_size)
        offspring = []
        for i in range(0, population_size, 2):
            p1, p2 = parents[i], parents[i + 1] if i + 1 < population_size else parents[0]
            if random.random() < crossover_prob:
                c1 = order_crossover(p1, p2)
                c2 = order_crossover(p2, p1)
            else:
                c1, c2 = list(p1), list(p2)
            c1 = mutate(c1, swap_mut_prob, two_opt_mut_prob, 0.0)
            c2 = mutate(c2, swap_mut_prob, two_opt_mut_prob, 0.0)
            offspring.append(c1)
            if len(offspring) < population_size:
                offspring.append(c2)
        offspring = offspring[:population_size]
        off_fitnesses = [fitness_fn(g) for g in offspring]
        combined = list(zip(pop + offspring, fitnesses + off_fitnesses))
        combined.sort(key=lambda x: x[1], reverse=True)
        pop = [list(x[0]) for x in combined[:population_size]]
        fitnesses = [x[1] for x in combined[:population_size]]
        if fitnesses[0] > best_fitness:
            best_fitness = fitnesses[0]
            best_genotype = list(pop[0])
    return best_genotype, best_fitness
