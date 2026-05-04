"""
Experiment runner — benchmarks all HPO algorithms on YAHPO Gym scenarios.

Runs each optimizer on each benchmark for N_SEEDS independent trials,
tracking the incumbent (best-so-far) performance at each evaluation step.

Results are saved as pickled lists of dicts to ``results/``, with one
file per (optimizer, scenario, instance, seed) combination.

Usage:
    python experiment.py
"""

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

# =========================================
# Configuration
# =========================================

N_SEEDS = 10          # Number of independent runs per (algorithm, benchmark)
BUDGET_MULTIPLIER = 50  # total_budget = BUDGET_MULTIPLIER × max_budget

BENCHMARKS = [
    # (scenario,       instance, fidelity_param, metric)
    ("nb301",          "cifar10", "epoch",       "val_accuracy"),
    ("rbv2_xgboost",   "16",      "trainsize",   "acc"),
]

OPTIMISERS = [
    RandomSearch,
    GridSearch,
    BayesianOptimisation,
    SuccessiveHalving,
    Hyperband,
]


# =========================================
# Runner
# =========================================

def run(
    optimiser_class,
    scenario: str,
    instance: str,
    fidelity_param: str,
    metric: str,
    total_budget: float,
    min_budget: float,
    max_budget: float,
    seed: int = 0,
) -> list[dict]:
    """
    Run a single (optimiser, benchmark, seed) experiment.

    Parameters
    ----------
    optimiser_class : type
        One of the HPOAlgorithm subclasses.
    scenario, instance : str
        YAHPO Gym benchmark identifiers.
    fidelity_param : str
        Name of the fidelity hyperparameter (e.g. "epoch", "trainsize").
    metric : str
        Name of the metric to optimise (e.g. "val_accuracy", "acc").
    total_budget : float
        Total fidelity budget (in fidelity units).
    min_budget, max_budget : float
        Fidelity bounds from YAHPO's fidelity space.
    seed : int
        Random seed for this run.

    Returns
    -------
    list[dict]
        List of evaluation records, each containing:
        - 'config': dict of hyperparameter values
        - 'result': float metric value
        - 'eval_budget': float fidelity consumed by this evaluation
        - 'cumulative_budget': float total fidelity consumed so far
    """
    # Set up the benchmark
    bench = BenchmarkSet(scenario)
    bench.set_instance(instance)

    # Get the HP search space (without fidelity parameters)
    cs = bench.get_opt_space(drop_fidelity_params=True)

    # YAHPO's objective_function needs ALL params (including fidelity +
    # internal params like 'repl'). Discover which params were dropped
    # so we can add sensible defaults when calling objective_function.
    cs_full = bench.get_opt_space(drop_fidelity_params=False)
    dropped_hps = {}
    full_names = {hp.name for hp in cs_full.get_hyperparameters()}
    opt_names = {hp.name for hp in cs.get_hyperparameters()}
    for name in full_names - opt_names:
        hp = cs_full.get_hyperparameter(name)
        if hasattr(hp, 'lower'):
            dropped_hps[name] = hp.lower  # default to lower bound
        elif hasattr(hp, 'value'):
            dropped_hps[name] = hp.value

    # Determine the expected type for the fidelity parameter
    fid_hp = cs_full.get_hyperparameter(fidelity_param)
    fid_is_int = isinstance(fid_hp, UniformIntegerHyperparameter)

    # Instantiate the optimiser
    optimiser = optimiser_class(
        cs, total_budget, min_budget, max_budget, seed=seed
    )

    runs = []
    cur_budget = 0.0

    while cur_budget < total_budget:
        # Get the next configuration and its evaluation budget
        config, eval_budget = optimiser.ask()

        # Cast fidelity to the correct type (nb301 epoch = int, xgboost trainsize = float)
        # Also clamp to [min_budget, max_budget] to prevent overshoot from SH/HB rung math
        eval_budget = min(eval_budget, max_budget)
        eval_budget = max(eval_budget, min_budget)
        if fid_is_int:
            eval_budget = int(round(eval_budget))

        # Evaluate on the YAHPO benchmark
        # YAHPO v1.0.2: fidelity + internal params merged into config dict
        config_dict = dict(config)
        config_dict.update(dropped_hps)         # add dropped params (repl, etc.)
        config_dict[fidelity_param] = eval_budget  # set fidelity to requested budget
        result_dict = bench.objective_function(config_dict)[0]
        result = result_dict[metric]

        # Update the optimiser with the result
        optimiser.tell(config, result, eval_budget)

        # Track cumulative budget
        cur_budget += eval_budget
        runs.append({
            "config": dict(config),       # save only HP values (no fidelity/repl)
            "result": float(result),
            "eval_budget": float(eval_budget),
            "cumulative_budget": float(cur_budget),
        })

    return runs


# =========================================
# Main
# =========================================

if __name__ == "__main__":
    os.makedirs("results", exist_ok=True)

    for scenario, instance, fidelity_param, metric in BENCHMARKS:
        # Discover min/max budget by diffing full vs dropped config spaces
        bench = BenchmarkSet(scenario)
        bench.set_instance(instance)
        cs_full = bench.get_opt_space(drop_fidelity_params=False)
        fid_hp = cs_full.get_hyperparameter(fidelity_param)
        min_budget = float(fid_hp.lower)
        max_budget = float(fid_hp.upper)
        total_budget = BUDGET_MULTIPLIER * max_budget

        print(f"\n{'='*60}")
        print(f"Benchmark: {scenario} / {instance}")
        print(f"  Fidelity: {fidelity_param}  [{min_budget} .. {max_budget}]")
        print(f"  Metric:   {metric} (maximize)")
        print(f"  Total budget: {total_budget}")
        print(f"{'='*60}")

        for optimiser_class in OPTIMISERS:
            for seed in range(N_SEEDS):
                fname = (
                    f"results/{optimiser_class.__name__}"
                    f"_{scenario}_{instance}_seed{seed}.pkl"
                )

                # Skip if already computed (allows resuming)
                if os.path.exists(fname):
                    print(f"  [SKIP] {fname} (already exists)")
                    continue

                print(
                    f"  Running {optimiser_class.__name__} "
                    f"seed={seed} ...",
                    end="",
                    flush=True,
                )

                # NOTE: all HPO algorithms use np.random.RandomState(seed),
                # not the global numpy RNG.  np.random.seed() has no effect
                # on their behaviour; it is kept here only to guard any
                # third-party code that may rely on the global state.
                np.random.seed(seed)
                results = run(
                    optimiser_class,
                    scenario,
                    instance,
                    fidelity_param,
                    metric,
                    total_budget,
                    min_budget,
                    max_budget,
                    seed=seed,
                )

                with open(fname, "wb") as f:
                    pickle.dump(results, f)

                # Report best result for this run
                best = max(r["result"] for r in results)
                print(f"  done  (best {metric}={best:.4f}, {len(results)} evals)")

    print("\n✓ All experiments complete.")
