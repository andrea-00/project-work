"""
Solver gerarchico: partition → Meta-GA → Intra-GA → stitch.
Restituisce la permutazione completa e il costo (via simulate_tour).
"""

from typing import List, Dict, Optional
import random

from problem_base import Problem
from .gold_problem_adapter import GoldCollectionAdapter
from .simulator import simulate_tour
from .partition import partition
from .linker import stitch
from .factory import (
    heuristic_intra_tours,
    meta_fitness_builder,
    compute_entry_loads_and_nodes,
)
from .ga import run_ga
from .return_optimizer import hill_climbing_returns, simulated_annealing_returns


def meta_initializer(k: int):
    """Genera permutazioni di [1, 2, ..., k-1] per il meta-GA."""
    def init():
        return random.sample(range(1, k), k - 1)
    return init


def hierarchical_solve(
    problem: Problem,
    k: int,
    *,
    meta_generations: int = 50,
    meta_pop_size: int = 30,
    intra_generations: int = 30,
    intra_pop_size: int = 20,
    optimize_returns: bool = True,
    return_optimizer: str = "hc",
    return_max_iters: int = 80,
    return_start_all_true: bool = True,
    rng_seed: Optional[int] = None,
) -> tuple[List[int], float]:
    """
    Risolve il problema con strategia gerarchica (Fase 2–5).

    Args:
        problem: istanza Problem.
        k: numero di cluster (cluster 0 = {0}, resto partizionato in k-1).
        meta_generations, meta_pop_size: parametri Meta-GA.
        intra_generations, intra_pop_size: parametri Intra-GA per cluster.
        optimize_returns: se True, ottimizza i ritorni a 0 (Fase 5) con HC o SA.
        return_optimizer: "hc", "sa", o "both" (esegue HC e SA e tiene il migliore).
        return_max_iters: iterazioni massime per HC/SA sui bit di ritorno.
        return_start_all_true: True = partenza "ritorna dopo ogni cluster" (beta>1);
            False = partenza "nessun ritorno" (beta<1, tour lunghi).
        rng_seed: seed per riproducibilità.

    Returns:
        (full_permutation, total_cost).
    """
    if rng_seed is not None:
        random.seed(rng_seed)

    adapter = GoldCollectionAdapter(problem)
    metric = adapter.get_cost_metric()
    n = adapter.num_elements

    if k < 2:
        # Un solo cluster: ordine NN su tutti i nodi
        order = [0] + list(range(1, n))
        cost = simulate_tour(problem, order, start_at_base=True, end_at_base=True)
        return order, cost

    # --- Step 1: Partition ---
    cluster_map = partition(adapter, k)

    # --- Step 2: Meta-GA (ordine cluster) ---
    meta_fitness = meta_fitness_builder(problem, adapter, cluster_map, k)
    meta_genotype, _ = run_ga(
        meta_fitness,
        meta_initializer(k),
        n_generations=meta_generations,
        population_size=meta_pop_size,
        rng_seed=rng_seed,
    )
    cluster_order = [0] + meta_genotype  # [0, c1, c2, ..., c_{k-1}]

    # --- Step 3: Euristica intra per avere intra_solutions (per entry_load/link) ---
    intra_heuristic = heuristic_intra_tours(adapter, cluster_map, metric, problem)

    # --- Step 4: Entry loads e entry nodes per ogni cluster ---
    entry_loads, entry_nodes, _ = compute_entry_loads_and_nodes(
        problem, cluster_order, intra_heuristic, metric, cluster_map
    )

    # --- Step 5: Intra-GA per ogni cluster (eccetto 0) ---
    intra_solutions: Dict[int, List[int]] = {0: [0]}
    for cid in range(1, k):
        if cid not in cluster_map or not cluster_map[cid]:
            continue
        nodes = cluster_map[cid]
        entry_n = entry_nodes.get(cid, nodes[0])
        entry_load = entry_loads.get(cid, 0.0)
        # Sub-adapter con entry_load e entry_node
        index_mapping = {i: nodes[i] for i in range(len(nodes))}
        sub = adapter.create_sub_problem(
            nodes,
            index_mapping,
            entry_load=entry_load,
            entry_node=entry_n,
        )
        # Inizializzatore: permutazione dei nodi del cluster (indici locali 0..len-1)
        size = len(nodes)
        def init_intra():
            return random.sample(range(size), size)
        intra_genotype, _ = run_ga(
            sub.get_fitness_function(),
            init_intra,
            n_generations=intra_generations,
            population_size=min(intra_pop_size, max(2, len(nodes) * 2)),
            rng_seed=rng_seed,
        )
        # Traduci in indici originali
        intra_solutions[cid] = [nodes[i] for i in intra_genotype]

    # --- Step 6: Ottimizzazione ritorni a 0 (Fase 5) ---
    if optimize_returns and k >= 2:
        if return_optimizer == "both":
            bits_hc, cost_hc = hill_climbing_returns(
                problem, cluster_order, intra_solutions, metric,
                max_iters=return_max_iters,
                start_all_true=return_start_all_true,
                rng_seed=rng_seed,
            )
            bits_sa, cost_sa = simulated_annealing_returns(
                problem, cluster_order, intra_solutions, metric,
                max_iters=return_max_iters * 2,
                start_all_true=return_start_all_true,
                rng_seed=(rng_seed + 1) if rng_seed is not None else None,
            )
            return_bits, cost = (bits_sa, cost_sa) if cost_sa < cost_hc else (bits_hc, cost_hc)
        elif return_optimizer == "sa":
            return_bits, cost = simulated_annealing_returns(
                problem, cluster_order, intra_solutions, metric,
                max_iters=return_max_iters,
                start_all_true=return_start_all_true,
                rng_seed=rng_seed,
            )
        else:
            return_bits, cost = hill_climbing_returns(
                problem, cluster_order, intra_solutions, metric,
                max_iters=return_max_iters,
                start_all_true=return_start_all_true,
                rng_seed=rng_seed,
            )
        full_permutation = stitch(
            cluster_order, intra_solutions, metric,
            return_after_cluster=return_bits,
        )
    else:
        full_permutation = stitch(cluster_order, intra_solutions, metric)
        cost = simulate_tour(
            problem, full_permutation, start_at_base=True, end_at_base=True
        )
    return full_permutation, cost
