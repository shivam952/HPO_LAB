"""
Verify that every number in the final report is reproducible.
Run after experiment.py has completed all seeds.
"""
import os
import glob
import pickle
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(BASE_DIR, "results")

BENCHMARKS = [
    ("nb301",        "cifar10", "val_accuracy", "NB301 (CIFAR-10)"),
    ("rbv2_xgboost", "16",      "acc",          "rbv2_xgboost Dataset 16"),
    ("rbv2_svm",     "16",      "acc",          "rbv2_svm Dataset 16"),
    ("rbv2_glmnet",  "16",      "acc",          "rbv2_glmnet Dataset 16"),
    ("rbv2_ranger",  "16",      "acc",          "rbv2_ranger Dataset 16"),
    ("lcbench",      "3945",    "val_accuracy", "LCBench Dataset 3945"),
]

ALGOS = [
    "RandomSearch", "GridSearch", "SuccessiveHalving",
    "BayesianOptimisation", "Hyperband", "BOHB_Round", "BOHB_Bracket",
]
N_SEEDS_EXPECTED = 10


def load(scenario, instance):
    data = {}
    for fpath in sorted(glob.glob(f"{RESULTS_DIR}/*_{scenario}_{instance}_seed*.pkl")):
        algo = os.path.basename(fpath).split(f"_{scenario}_{instance}")[0]
        with open(fpath, "rb") as f:
            data.setdefault(algo, []).append(pickle.load(f))
    return data


def incumbent(runs):
    budgets = np.array([r["cumulative_budget"] for r in runs])
    incs = np.maximum.accumulate([r["result"] for r in runs])
    return budgets, incs


print("=" * 70)
print("  FINAL PERFORMANCE  (mean ± std over seeds)")
print("=" * 70)

for scenario, instance, metric, title in BENCHMARKS:
    data = load(scenario, instance)
    print(f"\n  {title}  [{metric}]")
    print(f"  {'Algorithm':<25} {'Mean':>8} {'Std':>8} {'Min':>8} {'Max':>8} {'Seeds':>6}")
    print(f"  {'-'*66}")
    for algo in ALGOS:
        if algo not in data:
            print(f"  {algo:<25} MISSING")
            continue
        finals = [incumbent(r)[1][-1] for r in data[algo]]
        v = np.array(finals)
        print(f"  {algo:<25} {v.mean():8.4f} {v.std():8.4f} "
              f"{v.min():8.4f} {v.max():8.4f} {len(v):6d}")

print()
print("=" * 70)
print("  SEED COUNT CHECK")
print("=" * 70)
for scenario, instance, metric, title in BENCHMARKS:
    data = load(scenario, instance)
    print(f"\n  {title}")
    all_ok = True
    for algo in ALGOS:
        n = len(data.get(algo, []))
        status = "OK" if n == N_SEEDS_EXPECTED else f"WARNING: expected {N_SEEDS_EXPECTED}"
        if n != N_SEEDS_EXPECTED:
            all_ok = False
        print(f"    {algo:<25} {n:3d} seeds  {status}")
    if all_ok:
        print(f"    All {len(ALGOS)} algorithms: {N_SEEDS_EXPECTED} seeds each  ✓")

print("\nDone.")
