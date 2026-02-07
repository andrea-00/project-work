from problem_base import Problem

BASE_CONFIGS = [
    (100, 0.2, 1.0, 0.5),
    (100, 0.2, 1.0, 1.0),
    (100, 0.2, 1.0, 2.0),
    (100, 0.2, 2.0, 1.0),
    (100, 0.2, 2.0, 2.0),
    (100, 1.0, 1.0, 0.5),
    (100, 1.0, 1.0, 1.0),
    (100, 1.0, 1.0, 2.0),
    (100, 1.0, 2.0, 1.0),
    (100, 1.0, 2.0, 2.0),
    (1000, 0.2, 1.0, 0.5),
    (1000, 0.2, 1.0, 1.0),
    (1000, 0.2, 1.0, 2.0),
    (1000, 1.0, 1.0, 1.0),
    (1000, 1.0, 2.0, 1.0),
    (1000, 1.0, 1.0, 2.0),
]

HARD_CONFIGS = [
    (2000, 0.2, 1.0, 1.0),
    (2000, 0.5, 1.0, 1.0),
    (2000, 0.2, 2.0, 1.0),
    (5000, 0.2, 1.0, 1.0),
]


def get_instance_configs(n_seeds=10, include_hard=False):
    configs = list(BASE_CONFIGS)
    if include_hard:
        configs = configs + list(HARD_CONFIGS)
    for num_cities, density, alpha, beta in configs:
        for seed in range(n_seeds):
            yield (num_cities, density, alpha, beta, seed)


def build_problem(num_cities, density, alpha, beta, seed):
    return Problem(num_cities, alpha=alpha, beta=beta, density=density, seed=seed)
