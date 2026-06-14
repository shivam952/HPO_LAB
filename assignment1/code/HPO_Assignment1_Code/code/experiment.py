"""Run all HPO experiments and save results to results/."""

import os
import pickle

import numpy as np
from yahpo_gym import BenchmarkSet
from ConfigSpace.hyperparameters import UniformIntegerHyperparameter

from random_search import RandomSearch
from bayesian_optimisation import BayesianOptimisation
from grid_search import GridSearch
from successive_halving import SuccessiveHalving
from hyperband import Hyperband

N_SEEDS = 10
BUDGET_MULTIPLIER = 50

BENCHMARKS = [
    ("nb301",        "cifar10", "epoch",     "val_accuracy"),
    ("rbv2_xgboost", "16",      "trainsize", "acc"),
]

OPTIMISERS = [RandomSearch, GridSearch, BayesianOptimisation, SuccessiveHalving, Hyperband]


def run(optimiser_class, scenario, instance, fidelity_param, metric,
        total_budget, min_budget, max_budget, seed=0):
    bench = BenchmarkSet(scenario)
    bench.set_instance(instance)

    cs = bench.get_opt_space(drop_fidelity_params=True)
    cs_full = bench.get_opt_space(drop_fidelity_params=False)

    # Identify parameters dropped by drop_fidelity_params=True (e.g. the fidelity
    # param itself and replication count 'repl' in rbv2 benchmarks).
    # We fix them at their lower bound, which is the natural "cheapest" default:
    # repl=1 means a single replication (no averaging), consistent with the
    # surrogate's training conditions and the YAHPO Gym documentation.
    opt_names = {hp.name for hp in cs.get_hyperparameters()}
    dropped_hps = {}
    for hp in cs_full.get_hyperparameters():
        if hp.name not in opt_names:
            dropped_hps[hp.name] = hp.lower if hasattr(hp, 'lower') else hp.value

    fid_hp = cs_full.get_hyperparameter(fidelity_param)
    fid_is_int = isinstance(fid_hp, UniformIntegerHyperparameter)

    optimiser = optimiser_class(cs, total_budget, min_budget, max_budget, seed=seed)

    runs = []
    cur_budget = 0.0

    while cur_budget < total_budget:
        config, eval_budget = optimiser.ask()

        eval_budget = float(np.clip(eval_budget, min_budget, max_budget))
        if fid_is_int:
            eval_budget = int(round(eval_budget))

        config_dict = dict(config)
        config_dict.update(dropped_hps)
        config_dict[fidelity_param] = eval_budget
        result = bench.objective_function(config_dict)[0][metric]

        optimiser.tell(config, result, eval_budget)
        cur_budget += eval_budget
        runs.append({
            "config": dict(config),
            "result": float(result),
            "eval_budget": float(eval_budget),
            "cumulative_budget": float(cur_budget),
        })

    return runs


if __name__ == "__main__":
    os.makedirs("results", exist_ok=True)

    for scenario, instance, fidelity_param, metric in BENCHMARKS:
        bench = BenchmarkSet(scenario)
        bench.set_instance(instance)
        fid_hp = bench.get_opt_space(drop_fidelity_params=False).get_hyperparameter(fidelity_param)
        min_budget = float(fid_hp.lower)
        max_budget = float(fid_hp.upper)
        total_budget = BUDGET_MULTIPLIER * max_budget

        print(f"\n{'='*60}")
        print(f"Benchmark: {scenario} / {instance}")
        print(f"  Fidelity: {fidelity_param}  [{min_budget} .. {max_budget}]")
        print(f"  Total budget: {total_budget}")
        print(f"{'='*60}")

        for optimiser_class in OPTIMISERS:
            for seed in range(N_SEEDS):
                fname = f"results/{optimiser_class.__name__}_{scenario}_{instance}_seed{seed}.pkl"
                if os.path.exists(fname):
                    print(f"  [SKIP] {fname}")
                    continue

                print(f"  Running {optimiser_class.__name__} seed={seed} ...", end="", flush=True)
                np.random.seed(seed)
                results = run(optimiser_class, scenario, instance, fidelity_param,
                              metric, total_budget, min_budget, max_budget, seed=seed)

                with open(fname, "wb") as f:
                    pickle.dump(results, f)

                best = max(r["result"] for r in results)
                print(f"  done  (best={best:.4f}, {len(results)} evals)")

    print("\nAll experiments complete.")
