import sys
from instance_gen import build_problem
from strategy import solution


def main():
    n_cities = int(sys.argv[1]) if len(sys.argv) > 1 else 50
    n_seeds = int(sys.argv[2]) if len(sys.argv) > 2 else 2
    regimes = [
        (0.5, "beta < 1 (sub-lineare)"),
        (1.0, "beta = 1 (lineare)"),
        (2.0, "beta > 1 (super-lineare)"),
    ]
    print(f"Test regimi beta | n={n_cities} seeds={n_seeds} density=0.5 alpha=1")
    print("-" * 70)
    print(f"{'Regime':<28} {'Baseline':>12} {'Strategy':>12} {'Ratio':>8} {'Improve%':>10}")
    print("-" * 70)
    for beta, label in regimes:
        base_sum = 0.0
        strat_sum = 0.0
        for seed in range(n_seeds):
            P = build_problem(n_cities, 0.5, 1.0, beta, seed=seed)
            base_sum += P.baseline()
            strat_sum += solution(P)
        base_mean = base_sum / n_seeds
        strat_mean = strat_sum / n_seeds
        ratio = base_mean / strat_mean if strat_mean > 0 else 0
        improve_pct = 100 * (base_mean - strat_mean) / base_mean if base_mean > 0 else 0
        print(f"{label:<28} {base_mean:>12.0f} {strat_mean:>12.0f} {ratio:>8.3f} {improve_pct:>9.1f}%")
    print("-" * 70)


if __name__ == "__main__":
    main()
