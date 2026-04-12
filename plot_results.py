"""
Plotting script for HPO Assignment 1 results.

Produces two main figures for the report:
  1. Incumbent curves — mean ± std across seeds, one subplot per benchmark
  2. Final performance box plots — distribution at end of budget per algorithm

Usage:
    python plot_results.py

Reads results from results/ directory (pickled by experiment.py).
Saves figures to figures/ directory.
"""

import os
import glob
import pickle
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

RESULTS_DIR = "results"
FIGURES_DIR = "figures"

BENCHMARKS = [
    ("nb301", "cifar10", "val_accuracy", "NB301 (CIFAR10)"),
    ("rbv2_xgboost", "16", "acc", "rbv2_xgboost (Dataset 16)"),
]

# Algorithm display order and colours — consistent across all plots
ALGO_CONFIG = {
    "RandomSearch":        {"label": "Random Search",        "color": "#4C72B0", "ls": "-"},
    "GridSearch":          {"label": "Grid Search",          "color": "#DD8452", "ls": "-"},
    "SuccessiveHalving":   {"label": "Successive Halving",   "color": "#55A868", "ls": "-"},
    "Hyperband":           {"label": "Hyperband",            "color": "#C44E52", "ls": "-"},
    "BayesianOptimisation": {"label": "Bayesian Optimisation", "color": "#8172B3", "ls": "-"},
}

ALGO_ORDER = list(ALGO_CONFIG.keys())

# Number of interpolation points for incumbent curves
N_INTERP = 500


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_results(scenario: str, instance: str) -> dict[str, list[list[dict]]]:
    """
    Load all result files for a given (scenario, instance).

    Returns
    -------
    dict mapping algorithm name → list of seed runs,
    where each seed run is a list of evaluation dicts.
    """
    results = defaultdict(list)
    pattern = os.path.join(RESULTS_DIR, f"*_{scenario}_{instance}_seed*.pkl")
    files = sorted(glob.glob(pattern))

    for fpath in files:
        fname = os.path.basename(fpath)
        # Parse: AlgoName_scenario_instance_seedN.pkl
        # Algorithm name is everything before the first _scenario
        algo_name = fname.split(f"_{scenario}")[0]
        with open(fpath, "rb") as f:
            runs = pickle.load(f)
        results[algo_name].append(runs)

    return dict(results)


def compute_incumbents(runs: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute the incumbent curve from a single seed run.

    Returns (budgets, incumbents) arrays where incumbents[i] is the
    best result found using cumulative budget <= budgets[i].
    """
    budgets = np.array([r["cumulative_budget"] for r in runs])
    values = np.array([r["result"] for r in runs])
    incumbents = np.maximum.accumulate(values)  # maximize
    return budgets, incumbents


def interpolate_incumbents(
    all_budgets: list[np.ndarray],
    all_incumbents: list[np.ndarray],
    n_points: int = N_INTERP,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Interpolate incumbent curves to a common x-axis, then compute
    mean ± std across seeds.

    Returns (x_common, mean, std).
    """
    # Common x-axis: from the minimum starting budget to the max ending budget
    x_min = min(b[0] for b in all_budgets)
    x_max = max(b[-1] for b in all_budgets)
    x_common = np.linspace(x_min, x_max, n_points)

    # Interpolate each seed onto the common x-axis (step function: hold last value)
    interp_values = []
    for budgets, incumbents in zip(all_budgets, all_incumbents):
        # np.interp with left= fills before first observation with that value
        interp = np.interp(x_common, budgets, incumbents,
                           left=incumbents[0])
        interp_values.append(interp)

    interp_values = np.array(interp_values)  # (n_seeds, n_points)
    mean = np.mean(interp_values, axis=0)
    std = np.std(interp_values, axis=0)

    return x_common, mean, std


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_incumbent_curves(save_path: str = None):
    """
    Figure 1: Incumbent curves — one subplot per benchmark, all algorithms
    overlaid with mean ± 1 std shaded.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax, (scenario, instance, metric, title) in zip(axes, BENCHMARKS):
        results = load_results(scenario, instance)

        for algo_name in ALGO_ORDER:
            if algo_name not in results:
                continue
            cfg = ALGO_CONFIG[algo_name]
            seed_runs = results[algo_name]

            all_budgets = []
            all_incumbents = []
            for runs in seed_runs:
                b, inc = compute_incumbents(runs)
                all_budgets.append(b)
                all_incumbents.append(inc)

            x, mean, std = interpolate_incumbents(all_budgets, all_incumbents)

            ax.plot(x, mean, label=cfg["label"], color=cfg["color"],
                    ls=cfg["ls"], linewidth=1.8)
            ax.fill_between(x, mean - std, mean + std,
                            alpha=0.15, color=cfg["color"])

        ax.set_xlabel("Cumulative Budget (fidelity units)")
        ax.set_ylabel(metric)
        ax.set_title(title)
        ax.legend(fontsize=8, loc="lower right")
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"  Saved: {save_path}")
    plt.close()


def plot_final_performance(save_path: str = None):
    """
    Figure 2: Final performance box plot — distribution of best-found
    metric at the end of budget, per algorithm.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax, (scenario, instance, metric, title) in zip(axes, BENCHMARKS):
        results = load_results(scenario, instance)

        labels = []
        data = []
        colors = []

        for algo_name in ALGO_ORDER:
            if algo_name not in results:
                continue
            cfg = ALGO_CONFIG[algo_name]
            seed_runs = results[algo_name]

            # Best result per seed = last incumbent value
            final_values = []
            for runs in seed_runs:
                _, inc = compute_incumbents(runs)
                final_values.append(inc[-1])

            labels.append(cfg["label"])
            data.append(final_values)
            colors.append(cfg["color"])

        bp = ax.boxplot(data, labels=labels, patch_artist=True,
                        widths=0.6, showmeans=True,
                        meanprops=dict(marker='D', markerfacecolor='white',
                                       markeredgecolor='black', markersize=5))

        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)

        ax.set_ylabel(metric)
        ax.set_title(title)
        ax.tick_params(axis='x', rotation=30)
        ax.grid(True, axis='y', alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"  Saved: {save_path}")
    plt.close()


def plot_summary_table():
    """
    Print a text summary table of final performance (mean ± std) per algorithm.
    """
    print("\n" + "=" * 70)
    print("  Final Performance Summary (mean ± std across seeds)")
    print("=" * 70)

    for scenario, instance, metric, title in BENCHMARKS:
        results = load_results(scenario, instance)
        print(f"\n  {title}  ({metric})")
        print(f"  {'Algorithm':<25s}  {'Mean':>8s}  {'± Std':>8s}  {'Best':>8s}  {'Seeds':>5s}")
        print(f"  {'-'*60}")

        for algo_name in ALGO_ORDER:
            if algo_name not in results:
                continue
            cfg = ALGO_CONFIG[algo_name]
            seed_runs = results[algo_name]

            final_values = []
            for runs in seed_runs:
                _, inc = compute_incumbents(runs)
                final_values.append(inc[-1])

            vals = np.array(final_values)
            print(f"  {cfg['label']:<25s}  {vals.mean():8.4f}  {vals.std():8.4f}  "
                  f"{vals.max():8.4f}  {len(vals):5d}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    os.makedirs(FIGURES_DIR, exist_ok=True)

    print("Generating plots from results/...\n")

    # Check what's available
    all_files = glob.glob(os.path.join(RESULTS_DIR, "*.pkl"))
    if not all_files:
        print("  No result files found in results/. Run experiment.py first.")
        exit(1)

    print(f"  Found {len(all_files)} result files.\n")

    # Figure 1: Incumbent curves
    print("Figure 1: Incumbent Curves")
    plot_incumbent_curves(save_path=os.path.join(FIGURES_DIR, "incumbent_curves.pdf"))

    # Figure 2: Final performance
    print("Figure 2: Final Performance Box Plots")
    plot_final_performance(save_path=os.path.join(FIGURES_DIR, "final_performance.pdf"))

    # Text summary
    plot_summary_table()

    print(f"\n✓ All figures saved to {FIGURES_DIR}/")
