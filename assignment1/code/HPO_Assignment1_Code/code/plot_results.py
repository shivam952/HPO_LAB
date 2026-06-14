"""Generate incumbent curves and final performance box plots from results/."""

import os
import glob
import pickle
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

RESULTS_DIR = "results"
FIGURES_DIR = "figures"

BENCHMARKS = [
    ("nb301",        "cifar10", "val_accuracy", "NB301 (CIFAR10)"),
    ("rbv2_xgboost", "16",      "acc",          "rbv2_xgboost (Dataset 16)"),
]

ALGO_CONFIG = {
    "RandomSearch":          {"label": "Random Search",          "color": "#4C72B0", "ls": "-"},
    "GridSearch":            {"label": "Grid Search",            "color": "#DD8452", "ls": "-"},
    "SuccessiveHalving":     {"label": "Successive Halving",     "color": "#55A868", "ls": "-"},
    "Hyperband":             {"label": "Hyperband",              "color": "#C44E52", "ls": "-"},
    "BayesianOptimisation":  {"label": "Bayesian Optimisation",  "color": "#8172B3", "ls": "-"},
}

ALGO_ORDER = list(ALGO_CONFIG.keys())
N_INTERP = 500


def load_results(scenario, instance):
    results = defaultdict(list)
    for fpath in sorted(glob.glob(os.path.join(RESULTS_DIR, f"*_{scenario}_{instance}_seed*.pkl"))):
        algo_name = os.path.basename(fpath).split(f"_{scenario}_{instance}")[0]
        with open(fpath, "rb") as f:
            results[algo_name].append(pickle.load(f))
    return dict(results)


def compute_incumbents(runs):
    budgets = np.array([r["cumulative_budget"] for r in runs])
    incumbents = np.maximum.accumulate([r["result"] for r in runs])
    return budgets, incumbents


def interpolate_incumbents(all_budgets, all_incumbents):
    x = np.linspace(min(b[0] for b in all_budgets), max(b[-1] for b in all_budgets), N_INTERP)
    interp = np.array([np.interp(x, b, inc, left=inc[0])
                       for b, inc in zip(all_budgets, all_incumbents)])
    return x, interp.mean(axis=0), interp.std(axis=0)


def plot_incumbent_curves(save_path=None):
    fig, axes = plt.subplots(2, 1, figsize=(7, 10))
    for ax, (scenario, instance, metric, title) in zip(axes, BENCHMARKS):
        results = load_results(scenario, instance)
        for algo in ALGO_ORDER:
            if algo not in results:
                continue
            cfg = ALGO_CONFIG[algo]
            budgets, incumbents = zip(*[compute_incumbents(r) for r in results[algo]])
            x, mean, std = interpolate_incumbents(list(budgets), list(incumbents))
            ax.plot(x, mean, label=cfg["label"], color=cfg["color"], ls=cfg["ls"], linewidth=1.8)
            ax.fill_between(x, mean - std, mean + std, alpha=0.15, color=cfg["color"])
        ax.set_xlabel("Cumulative Budget (fidelity units)")
        ax.set_ylabel(metric)
        ax.set_title(title)
        ax.legend(fontsize=8, loc="lower right")
        ax.grid(True, alpha=0.3)
    plt.tight_layout(h_pad=2.0)
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"  Saved: {save_path}")
    plt.close()


def plot_final_performance(save_path=None):
    fig, axes = plt.subplots(2, 1, figsize=(7, 10))
    for ax, (scenario, instance, metric, title) in zip(axes, BENCHMARKS):
        results = load_results(scenario, instance)
        labels, data, colors = [], [], []
        for algo in ALGO_ORDER:
            if algo not in results:
                continue
            cfg = ALGO_CONFIG[algo]
            finals = [compute_incumbents(r)[1][-1] for r in results[algo]]
            labels.append(cfg["label"])
            data.append(finals)
            colors.append(cfg["color"])
        bp = ax.boxplot(data, labels=labels, patch_artist=True, widths=0.6, showmeans=True,
                        meanprops=dict(marker='D', markerfacecolor='white',
                                       markeredgecolor='black', markersize=5))
        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)
        ax.set_ylabel(metric)
        ax.set_title(title)
        ax.tick_params(axis='x', rotation=30)
        ax.grid(True, axis='y', alpha=0.3)
    plt.tight_layout(h_pad=2.0)
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"  Saved: {save_path}")
    plt.close()


def print_summary():
    print("\n" + "=" * 70)
    print("  Final Performance (mean ± std across seeds)")
    print("=" * 70)
    for scenario, instance, metric, title in BENCHMARKS:
        results = load_results(scenario, instance)
        print(f"\n  {title}  ({metric})")
        print(f"  {'Algorithm':<25}  {'Mean':>8}  {'±Std':>8}  {'Best':>8}  {'Seeds':>5}")
        print(f"  {'-'*60}")
        for algo in ALGO_ORDER:
            if algo not in results:
                continue
            vals = np.array([compute_incumbents(r)[1][-1] for r in results[algo]])
            print(f"  {ALGO_CONFIG[algo]['label']:<25}  {vals.mean():8.4f}  "
                  f"{vals.std():8.4f}  {vals.max():8.4f}  {len(vals):5d}")


if __name__ == "__main__":
    os.makedirs(FIGURES_DIR, exist_ok=True)
    files = glob.glob(os.path.join(RESULTS_DIR, "*.pkl"))
    if not files:
        print("No results found. Run experiment.py first.")
        exit(1)
    print(f"Found {len(files)} result files.\n")
    plot_incumbent_curves(save_path=os.path.join(FIGURES_DIR, "incumbent_curves.pdf"))
    plot_final_performance(save_path=os.path.join(FIGURES_DIR, "final_performance.pdf"))
    print_summary()
    print(f"\nFigures saved to {FIGURES_DIR}/")
