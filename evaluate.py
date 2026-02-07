import time
import numpy as np
from instance_gen import get_instance_configs, build_problem
from strategy import solution


def run_one(num_cities, density, alpha, beta, seed):
    P = build_problem(num_cities, density, alpha, beta, seed)
    t0 = time.perf_counter()
    baseline_cost = P.baseline()
    t1 = time.perf_counter()
    strategy_cost = solution(P)
    t2 = time.perf_counter()
    return {
        "baseline_cost": baseline_cost,
        "strategy_cost": strategy_cost,
        "baseline_time": t1 - t0,
        "strategy_time": t2 - t1,
        "config": (num_cities, density, alpha, beta, seed),
    }


def run_config(config_key, n_seeds=10):
    num_cities, density, alpha, beta = config_key
    results = []
    for seed in range(n_seeds):
        r = run_one(num_cities, density, alpha, beta, seed)
        results.append(r)
    baseline_costs = np.array([r["baseline_cost"] for r in results])
    strategy_costs = np.array([r["strategy_cost"] for r in results])
    baseline_times = np.array([r["baseline_time"] for r in results])
    strategy_times = np.array([r["strategy_time"] for r in results])
    return {
        "config": config_key,
        "baseline_cost_mean": float(np.mean(baseline_costs)),
        "baseline_cost_std": float(np.std(baseline_costs)),
        "strategy_cost_mean": float(np.mean(strategy_costs)),
        "strategy_cost_std": float(np.std(strategy_costs)),
        "baseline_time_mean": float(np.mean(baseline_times)),
        "strategy_time_mean": float(np.mean(strategy_times)),
        "improvement_ratio": float(np.mean(baseline_costs) / np.mean(strategy_costs))
        if np.mean(strategy_costs) > 0
        else float("inf"),
    }


def run_all(n_seeds=10, include_hard=False):
    configs_seen = set()
    configs_list = []
    for num_cities, density, alpha, beta, seed in get_instance_configs(
        n_seeds=n_seeds, include_hard=include_hard
    ):
        key = (num_cities, density, alpha, beta)
        if key not in configs_seen:
            configs_seen.add(key)
            configs_list.append(key)
    results = []
    for config_key in configs_list:
        results.append(run_config(config_key, n_seeds=n_seeds))
    return results


def main():
    n_seeds = 5
    results = run_all(n_seeds=n_seeds, include_hard=False)
    for r in results:
        c = r["config"]
        print(
            f"n={c[0]} d={c[1]} a={c[2]} b={c[3]}: "
            f"baseline {r['baseline_cost_mean']:.0f}±{r['baseline_cost_std']:.0f} "
            f"strategy {r['strategy_cost_mean']:.0f}±{r['strategy_cost_std']:.0f} "
            f"ratio={r['improvement_ratio']:.3f} "
            f"t_base={r['baseline_time_mean']:.2f}s t_str={r['strategy_time_mean']:.2f}s"
        )


if __name__ == "__main__":
    main()
