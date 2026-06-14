"""Generate incumbent curves and final performance box plots from results/."""

import os
import glob
import json
import pickle
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(BASE_DIR, "results")
FIGURES_DIR = os.path.join(BASE_DIR, "figures")
DIAG_DIR = os.path.join(RESULTS_DIR, "diagnostics")

BENCHMARKS = [
    ("nb301",        "cifar10", "val_accuracy", "NB301 (CIFAR-10)"),
    ("rbv2_xgboost", "16",      "acc",          "rbv2\\_xgboost (Dataset 16)"),
    ("rbv2_svm",     "16",      "acc",          "rbv2\\_svm (Dataset 16)"),
    ("rbv2_glmnet",  "16",      "acc",          "rbv2\\_glmnet (Dataset 16)"),
    ("rbv2_ranger",  "16",      "acc",          "rbv2\\_ranger (Dataset 16)"),
    ("lcbench",      "3945",    "val_accuracy", "LCBench (Dataset 3945)"),
]

ALGO_CONFIG = {
    "RandomSearch":         {"label": "Random Search",         "color": "#4C72B0", "ls": "-",  "lw": 1.8},
    "GridSearch":           {"label": "Grid Search",           "color": "#DD8452", "ls": "-",  "lw": 1.8},
    "SuccessiveHalving":    {"label": "Successive Halving",    "color": "#55A868", "ls": "-",  "lw": 1.8},
    "BayesianOptimisation": {"label": "Bayesian Opt.",         "color": "#8172B3", "ls": "-",  "lw": 1.8},
    "Hyperband":            {"label": "Hyperband",             "color": "#C44E52", "ls": "-",  "lw": 1.8},
    "BOHB_Round":           {"label": "BOHB-Round (ours)",     "color": "#1ABC9C", "ls": "--", "lw": 2.2},
    "BOHB_Bracket":         {"label": "BOHB-Bracket (ours)",   "color": "#E67E22", "ls": "--", "lw": 2.2},
}

ALGO_ORDER = list(ALGO_CONFIG.keys())
N_INTERP = 500
FONT_SIZE = 13


def load_results(scenario, instance):
    results = defaultdict(list)
    for fpath in sorted(glob.glob(os.path.join(RESULTS_DIR, f"*_{scenario}_{instance}_seed*.pkl"))):
        algo_name = os.path.basename(fpath).split(f"_{scenario}_{instance}")[0]
        with open(fpath, "rb") as f:
            results[algo_name].append(pickle.load(f))
    return dict(results)


def load_diagnostics(algo_name, scenario, instance):
    pattern = os.path.join(DIAG_DIR, f"{algo_name}_{scenario}_{instance}_seed*.json")
    runs = []
    for fpath in sorted(glob.glob(pattern)):
        with open(fpath, "r") as f:
            runs.append(json.load(f))
    return runs


def compute_incumbents(runs):
    budgets = np.array([r["cumulative_budget"] for r in runs])
    incumbents = np.maximum.accumulate([r["result"] for r in runs])
    return budgets, incumbents


def compute_regret(runs, global_best):
    values = np.array([r["result"] for r in runs])
    best_so_far = np.maximum.accumulate(values)
    regret = global_best - best_so_far
    return np.array([r["cumulative_budget"] for r in runs]), np.maximum(regret, 1e-12)


def interpolate_curves(all_budgets, all_curves):
    x_min = min(b[0] for b in all_budgets)
    x_max = max(b[-1] for b in all_budgets)
    x = np.linspace(x_min, x_max, N_INTERP)
    interp = np.array([
        np.interp(x, b, c, left=c[0])
        for b, c in zip(all_budgets, all_curves)
    ])
    median = np.median(interp, axis=0)
    q25 = np.percentile(interp, 25, axis=0)
    q75 = np.percentile(interp, 75, axis=0)
    return x, median, q25, q75


def _save(fig, name):
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(FIGURES_DIR, f"{name}.{ext}"), dpi=300, bbox_inches="tight")
    print(f"  Saved: {name}.[png/pdf]")


def plot_incumbent_curves():
    os.makedirs(FIGURES_DIR, exist_ok=True)
    for scenario, instance, metric, title in BENCHMARKS:
        results = load_results(scenario, instance)
        if not results:
            print(f"  [SKIP] No results for {scenario}/{instance}")
            continue

        global_best = max(
            max(r["result"] for r in runs)
            for algo_runs in results.values()
            for runs in algo_runs
        )

        fig, ax = plt.subplots(figsize=(8, 5))
        for algo in ALGO_ORDER:
            if algo not in results:
                continue
            cfg = ALGO_CONFIG[algo]
            budgets, regrets = zip(*[compute_regret(r, global_best) for r in results[algo]])
            x, median, q25, q75 = interpolate_curves(list(budgets), list(regrets))
            ax.plot(x, median, label=cfg["label"], color=cfg["color"],
                    ls=cfg["ls"], linewidth=cfg["lw"])
            ax.fill_between(x, q25, q75, alpha=0.15, color=cfg["color"])

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Cumulative Budget (fidelity units)", fontsize=FONT_SIZE)
        ax.set_ylabel("Regret", fontsize=FONT_SIZE)
        ax.set_title(title, fontsize=FONT_SIZE + 1)
        ax.legend(fontsize=FONT_SIZE - 2, loc="upper right")
        ax.tick_params(labelsize=FONT_SIZE - 2)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        _save(fig, f"{scenario}_{instance}_regret")
        plt.close(fig)


def plot_final_performance():
    os.makedirs(FIGURES_DIR, exist_ok=True)
    for scenario, instance, metric, title in BENCHMARKS:
        results = load_results(scenario, instance)
        if not results:
            continue

        labels, data, colors = [], [], []
        for algo in ALGO_ORDER:
            if algo not in results:
                continue
            cfg = ALGO_CONFIG[algo]
            finals = [compute_incumbents(r)[1][-1] for r in results[algo]]
            labels.append(cfg["label"])
            data.append(finals)
            colors.append(cfg["color"])

        fig, ax = plt.subplots(figsize=(9, 5))
        bp = ax.boxplot(data, labels=labels, patch_artist=True, widths=0.6,
                        showmeans=True,
                        meanprops=dict(marker='D', markerfacecolor='white',
                                       markeredgecolor='black', markersize=5))
        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)
        ax.set_ylabel(metric, fontsize=FONT_SIZE)
        ax.set_title(title, fontsize=FONT_SIZE + 1)
        ax.tick_params(axis='x', labelsize=FONT_SIZE - 2, rotation=30)
        ax.tick_params(axis='y', labelsize=FONT_SIZE - 2)
        ax.grid(True, axis="y", alpha=0.3)
        plt.tight_layout()
        _save(fig, f"{scenario}_{instance}_box")
        plt.close(fig)


def plot_surrogate_usage_ratio():
    """Cumulative model vs. random sampling ratio over the run (BOHB variants only)."""
    os.makedirs(FIGURES_DIR, exist_ok=True)
    for scenario, instance, metric, title in BENCHMARKS:
        fig, ax = plt.subplots(figsize=(8, 5))
        plotted = False
        for algo in ["BOHB_Round", "BOHB_Bracket"]:
            runs = load_diagnostics(algo, scenario, instance)
            if not runs:
                continue
            all_fit_ids, all_ratios = [], []
            for run in runs:
                history = run.get("fidelity_history", [])
                fit_ids = [e["fit_id"] for e in history]
                ratios = [
                    e["model_samples"] / max(1, e["model_samples"] + e["random_samples"])
                    for e in history
                ]
                all_fit_ids.append(np.asarray(fit_ids))
                all_ratios.append(np.asarray(ratios))
            if not all_fit_ids:
                continue
            x, median, q25, q75 = interpolate_curves(all_fit_ids, all_ratios)
            cfg = ALGO_CONFIG[algo]
            ax.plot(x, median, label=cfg["label"], color=cfg["color"],
                    ls=cfg["ls"], linewidth=cfg["lw"])
            ax.fill_between(x, q25, q75, alpha=0.15, color=cfg["color"])
            plotted = True

        if not plotted:
            plt.close(fig)
            continue
        ax.set_ylim(0, 1)
        ax.set_xlabel("Surrogate Fit ID", fontsize=FONT_SIZE)
        ax.set_ylabel("Model Usage Ratio", fontsize=FONT_SIZE)
        ax.set_title(title, fontsize=FONT_SIZE + 1)
        ax.legend(fontsize=FONT_SIZE - 2, loc="lower right")
        ax.tick_params(labelsize=FONT_SIZE - 2)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        _save(fig, f"{scenario}_{instance}_surrogate_usage_ratio")
        plt.close(fig)


def plot_surrogate_local_usage_ratio():
    """Per-interval model vs. random ratio (delta between consecutive fits)."""
    os.makedirs(FIGURES_DIR, exist_ok=True)
    for scenario, instance, metric, title in BENCHMARKS:
        fig, ax = plt.subplots(figsize=(8, 5))
        plotted = False
        for algo in ["BOHB_Round", "BOHB_Bracket"]:
            runs = load_diagnostics(algo, scenario, instance)
            if not runs:
                continue
            all_fit_ids, all_ratios = [], []
            for run in runs:
                history = run.get("fidelity_history", [])
                if len(history) < 2:
                    continue
                fit_ids, ratios = [], []
                prev_m = history[0]["model_samples"]
                prev_r = history[0]["random_samples"]
                for entry in history[1:]:
                    dm = entry["model_samples"] - prev_m
                    dr = entry["random_samples"] - prev_r
                    total = dm + dr
                    ratios.append(dm / total if total > 0 else 0.0)
                    fit_ids.append(entry["fit_id"])
                    prev_m = entry["model_samples"]
                    prev_r = entry["random_samples"]
                if fit_ids:
                    all_fit_ids.append(np.asarray(fit_ids))
                    all_ratios.append(np.asarray(ratios))
            if not all_fit_ids:
                continue
            x, median, q25, q75 = interpolate_curves(all_fit_ids, all_ratios)
            cfg = ALGO_CONFIG[algo]
            ax.plot(x, median, label=cfg["label"], color=cfg["color"],
                    ls=cfg["ls"], linewidth=cfg["lw"])
            ax.fill_between(x, q25, q75, alpha=0.15, color=cfg["color"])
            plotted = True

        if not plotted:
            plt.close(fig)
            continue
        ax.set_ylim(0, 1)
        ax.set_xlabel("Surrogate Fit ID", fontsize=FONT_SIZE)
        ax.set_ylabel("Local Model Usage Ratio", fontsize=FONT_SIZE)
        ax.set_title(title, fontsize=FONT_SIZE + 1)
        ax.legend(fontsize=FONT_SIZE - 2, loc="lower right")
        ax.tick_params(labelsize=FONT_SIZE - 2)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        _save(fig, f"{scenario}_{instance}_surrogate_local_usage_ratio")
        plt.close(fig)


def plot_b_star():
    """Evolution of selected training fidelity b* over the run (BOHB variants only)."""
    os.makedirs(FIGURES_DIR, exist_ok=True)
    for scenario, instance, metric, title in BENCHMARKS:
        fig, ax = plt.subplots(figsize=(8, 5))
        plotted = False
        for algo in ["BOHB_Round", "BOHB_Bracket"]:
            runs = load_diagnostics(algo, scenario, instance)
            if not runs:
                continue
            all_fit_ids, all_b_stars = [], []
            for run in runs:
                history = run.get("fidelity_history", [])
                if not history:
                    continue
                all_fit_ids.append(np.asarray([e["fit_id"] for e in history]))
                all_b_stars.append(np.asarray([e["b_star"] for e in history]))
            if not all_fit_ids:
                continue
            x, median, q25, q75 = interpolate_curves(all_fit_ids, all_b_stars)
            cfg = ALGO_CONFIG[algo]
            ax.plot(x, median, label=cfg["label"], color=cfg["color"],
                    ls=cfg["ls"], linewidth=cfg["lw"])
            ax.fill_between(x, q25, q75, alpha=0.15, color=cfg["color"])
            plotted = True

        if not plotted:
            plt.close(fig)
            continue
        ax.set_yscale("log")
        ax.set_xlabel("Surrogate Fit ID", fontsize=FONT_SIZE)
        ax.set_ylabel(r"Selected Fidelity $b^*$", fontsize=FONT_SIZE)
        ax.set_title(title, fontsize=FONT_SIZE + 1)
        ax.legend(fontsize=FONT_SIZE - 2, loc="lower right")
        ax.tick_params(labelsize=FONT_SIZE - 2)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        _save(fig, f"{scenario}_{instance}_b_star")
        plt.close(fig)


def print_surrogate_activation_summary():
    print("\n" + "=" * 70)
    print("  Surrogate activation (eval index at first fit)")
    print("=" * 70)
    for scenario, instance, metric, title in BENCHMARKS:
        print(f"\n  {title}  ({metric})")
        print(f"  {'Algorithm':<25}  {'Mean':>8}  {'±Std':>8}  {'Max':>8}  {'Seeds':>5}")
        print(f"  {'-' * 60}")
        for algo in ["BOHB_Round", "BOHB_Bracket"]:
            runs = load_diagnostics(algo, scenario, instance)
            acts = [
                r["surrogate_activation_eval"]
                for r in runs
                if r.get("surrogate_activation_eval") is not None
            ]
            if not acts:
                continue
            acts = np.asarray(acts)
            print(f"  {algo:<25}  {acts.mean():8.1f}  "
                  f"{acts.std():8.1f}  {acts.max():8.1f}  {len(acts):5d}")


def print_summary():
    print("\n" + "=" * 70)
    print("  Final Performance (mean ± std across seeds)")
    print("=" * 70)
    for scenario, instance, metric, title in BENCHMARKS:
        results = load_results(scenario, instance)
        if not results:
            continue
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
    plot_incumbent_curves()
    plot_final_performance()
    plot_surrogate_usage_ratio()
    plot_surrogate_local_usage_ratio()
    plot_b_star()
    print_summary()
    print_surrogate_activation_summary()
    print(f"\nFigures saved to {FIGURES_DIR}/")
