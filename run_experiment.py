import argparse
import sys
import time
from typing import Optional, Tuple

from instance_gen import build_problem
from hierarchical_gold.solver import hierarchical_solve
from hierarchical_gold.partition import partition
from hierarchical_gold.gold_problem_adapter import GoldCollectionAdapter


def _choose_k(num_cities: int, beta: float) -> int:
    if beta > 1.0:
        if num_cities <= 80:
            return num_cities
        if num_cities <= 200:
            return max(50, num_cities // 2)
        return max(80, min(150, num_cities // 5))
    if beta < 1.0:
        return max(2, min(12, num_cities // 12))
    if num_cities <= 40:
        return num_cities
    if num_cities <= 100:
        return max(2, min(15, num_cities // 8))
    return max(2, min(25, num_cities // 25))


def run_one(
    n_cities: int,
    density: float,
    alpha: float,
    beta: float,
    seed: int,
    k: Optional[int],
    meta_generations: int,
    meta_pop_size: int,
    intra_generations: int,
    intra_pop_size: int,
    return_optimizer: str,
    return_max_iters: int,
    return_start_all_true: Optional[bool],
) -> Tuple[float, float, float]:
    P = build_problem(n_cities, density, alpha, beta, seed=seed)
    base = P.baseline()
    k_use = k if k is not None else _choose_k(n_cities, beta)
    return_start = (beta >= 1.0) if return_start_all_true is None else return_start_all_true
    t0 = time.perf_counter()
    _, cost = hierarchical_solve(
        P,
        k_use,
        meta_generations=meta_generations,
        meta_pop_size=meta_pop_size,
        intra_generations=intra_generations,
        intra_pop_size=intra_pop_size,
        optimize_returns=True,
        return_optimizer=return_optimizer,
        return_max_iters=return_max_iters,
        return_start_all_true=return_start,
        rng_seed=seed,
    )
    elapsed = time.perf_counter() - t0
    return base, cost, elapsed


def main():
    ap = argparse.ArgumentParser(description="Run one experiment and print EXPERIMENTS.md block.")
    ap.add_argument("--n_cities", type=int, default=40, help="Number of cities")
    ap.add_argument("--n_seeds", type=int, default=2, help="Seeds per regime")
    ap.add_argument("--density", type=float, default=0.5)
    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--k", type=int, default=None, help="Override k (default: from beta)")
    ap.add_argument("--meta_gen", type=int, default=40)
    ap.add_argument("--meta_pop", type=int, default=25)
    ap.add_argument("--intra_gen", type=int, default=25)
    ap.add_argument("--intra_pop", type=int, default=15)
    ap.add_argument("--return_opt", choices=("hc", "sa", "both"), default="hc")
    ap.add_argument("--return_iters", type=int, default=80)
    ap.add_argument("--return_start_true", type=int, default=None, choices=(0, 1),
                    help="1=always True, 0=always False, omit=beta>=1")
    ap.add_argument("--name", type=str, default="", help="Short name for the attempt")
    args = ap.parse_args()
    return_start_all_true = None if args.return_start_true is None else bool(args.return_start_true)

    regimes = [(0.5, "beta<1"), (1.0, "beta=1"), (2.0, "beta>1")]
    lines = []
    lines.append("")
    lines.append(f"### Attempt – {args.name or 'run_experiment'}")
    lines.append("")
    lines.append("- **Parametri**:")
    lines.append(f"  - n_cities={args.n_cities}, n_seeds={args.n_seeds}, density={args.density}, alpha={args.alpha}")
    lines.append(f"  - k={args.k if args.k is not None else 'auto(beta)'}")
    lines.append(f"  - meta_generations={args.meta_gen}, meta_pop_size={args.meta_pop}")
    lines.append(f"  - intra_generations={args.intra_gen}, intra_pop_size={args.intra_pop}")
    lines.append(f"  - return_optimizer={args.return_opt}, return_max_iters={args.return_iters}")
    lines.append(f"  - return_start_all_true={return_start_all_true if return_start_all_true is not None else 'auto(beta>=1)'}")
    lines.append("- **Risultati**:")
    total_time = 0.0
    for beta, label in regimes:
        base_sum = strat_sum = time_sum = 0.0
        for seed in range(args.n_seeds):
            base, strat, elapsed = run_one(
                args.n_cities, args.density, args.alpha, beta, seed,
                args.k, args.meta_gen, args.meta_pop, args.intra_gen, args.intra_pop,
                args.return_opt, args.return_iters, return_start_all_true,
            )
            base_sum += base
            strat_sum += strat
            time_sum += elapsed
        total_time += time_sum
        base_mean = base_sum / args.n_seeds
        strat_mean = strat_sum / args.n_seeds
        ratio = base_mean / strat_mean if strat_mean > 0 else 0
        improve = 100 * (base_mean - strat_mean) / base_mean if base_mean > 0 else 0
        t_mean = time_sum / args.n_seeds
        lines.append(f"  - {label}: baseline={base_mean:.0f} strategy={strat_mean:.0f} ratio={ratio:.3f} improve%={improve:.1f}% time_mean={t_mean:.1f}s")
    lines.append(f"- **Tempo totale**: {total_time:.1f}s")
    lines.append("")
    for line in lines:
        print(line)
    return 0


if __name__ == "__main__":
    sys.exit(main())
