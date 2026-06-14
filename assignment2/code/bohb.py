"""BOHB: Bayesian Optimisation + Hyperband (Falkner et al., ICML 2018).

Surrogate: Random Forest (100 trees, min_samples_leaf=3) with Expected Improvement.
Encoding: label encoding for categoricals (one value per HP) rather than one-hot,
so that NB301's 34-dimensional space stays at d=34 instead of inflating to 272.

Two variants are provided:
  BOHB_Round   — surrogate is refitted once per round; all brackets share the same model.
  BOHB_Bracket — surrogate is refitted at the start of each bracket, so later brackets
                 benefit from observations accumulated during earlier ones in the same round.
"""
from __future__ import annotations

import json
import math
import os
from collections import deque
from typing import Dict, List, Optional, Tuple

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from ConfigSpace import Configuration, ConfigurationSpace

from hpo_algorithm import HPOAlgorithm
from utils import sample_config, config_to_label_array, config_key, _norm_cdf, _norm_pdf


class _RandomForestSurrogate:
    """RF surrogate: mean = average tree prediction, std = std across trees."""

    def __init__(self, n_estimators=100, min_samples_leaf=3, rng=None):
        self._n_estimators = n_estimators
        self._min_samples_leaf = min_samples_leaf
        self._rng = rng if rng is not None else np.random.RandomState(0)
        self._forest = None

    @property
    def is_fitted(self):
        return self._forest is not None

    def fit(self, X, y):
        seed = int(self._rng.randint(0, 2**31 - 1))
        self._forest = RandomForestRegressor(
            n_estimators=self._n_estimators,
            min_samples_leaf=self._min_samples_leaf,
            random_state=seed,
        )
        self._forest.fit(X, y)

    def predict(self, X):
        """Returns (mean, std) across individual tree predictions."""
        if not self.is_fitted:
            raise RuntimeError("Surrogate not fitted yet.")
        tree_preds = np.array([t.predict(X) for t in self._forest.estimators_])
        return tree_preds.mean(axis=0), tree_preds.std(axis=0)


def _expected_improvement(mean, std, best_f, xi=0.0):
    """EI(x) = (mu - f* - xi)*Phi(Z) + sigma*phi(Z),  Z = (mu - f* - xi)/sigma."""
    safe_std = np.where(std > 1e-10, std, 1e-10)
    Z = (mean - best_f - xi) / safe_std
    ei = (mean - best_f - xi) * _norm_cdf(Z) + std * _norm_pdf(Z)
    return np.where(std > 1e-10, ei, 0.0)


class _BOHBBracket:
    """Single Successive Halving bracket with a two-phase init: create then initialise."""

    def __init__(self, cs, n_configs, start_budget, max_budget, eta, rng):
        self._eta = eta
        self._rng = rng
        self._n_configs = n_configs

        # rung budget schedule: start_budget * eta^k, capped at max_budget
        self._rung_budgets = []
        b = start_budget
        while b <= max_budget + 1e-9:
            self._rung_budgets.append(float(min(b, max_budget)))
            b *= eta
        if not self._rung_budgets:
            self._rung_budgets = [max_budget]

        self._pending = deque()
        self._rung_idx = 0
        self._rung_configs = []
        self._results = {}
        self._exhausted = False
        self._initialised = False

    def initialise(self, config_sampler):
        """Draw initial configs using the provided sampler and enqueue them."""
        if self._initialised:
            raise RuntimeError("Bracket already initialised.")
        configs = [config_sampler() for _ in range(self._n_configs)]
        self._pending = deque((c, self._rung_budgets[0]) for c in configs)
        self._rung_configs = list(configs)
        self._initialised = True

    @property
    def exhausted(self):
        return self._exhausted and len(self._pending) == 0

    def ask(self):
        if not self._initialised:
            raise RuntimeError("Bracket not yet initialised.")
        if not self._pending:
            raise StopIteration
        return self._pending.popleft()

    def tell(self, config, result, budget):
        self._results[config_key(config)] = result

        if len(self._results) < len(self._rung_configs):
            return

        # promote top floor(n/eta) configs to the next rung
        n_promote = max(1, math.floor(len(self._rung_configs) / self._eta))
        ranked = sorted(self._rung_configs,
                        key=lambda c: self._results[config_key(c)], reverse=True)
        promoted = ranked[:n_promote]

        self._rung_idx += 1
        if self._rung_idx < len(self._rung_budgets):
            next_budget = self._rung_budgets[self._rung_idx]
            self._rung_configs = promoted
            self._results = {}
            for c in promoted:
                self._pending.append((c, next_budget))
        else:
            self._exhausted = True


class BOHB(HPOAlgorithm):
    """BOHB: Bayesian Optimisation + Hyperband (Falkner et al., ICML 2018).

    Runs s_max+1 Successive Halving brackets per round (identical to Hyperband).
    When enough observations are available at any fidelity level, a Random Forest
    surrogate is fitted and used to propose configs via EI. A constant fraction rho
    of configs is always sampled uniformly at random (Algorithm 2, line 1 of the paper),
    preserving convergence guarantees when low-fidelity evaluations are misleading.

    refit_mode controls when the surrogate is refitted:
      "round"   — once per round; all brackets share the same model snapshot.
      "bracket" — at the start of each bracket; later brackets benefit from earlier ones.

    N_min = max(10, n_hps + 1) following Algorithm 2 of the paper.
    """

    _N_MIN_FLOOR = 10

    def __init__(self, cs, total_budget, min_budget, max_budget,
                 eta=3, n_candidates=None, n_estimators=100,
                 min_samples_leaf=3, xi=0.0, rho=1/3, seed=0,
                 refit_mode="round"):
        super().__init__(cs, total_budget, min_budget, max_budget)
        self._eta = eta
        self._xi = xi
        self._rho = rho
        self._rng = np.random.RandomState(seed)
        self._cs = cs
        self._refit_mode = refit_mode

        # Hyperband geometry
        self._s_max = math.floor(math.log(max_budget / min_budget) / math.log(eta))
        self._B = (self._s_max + 1) * max_budget

        # n_hps: raw HP count — used for N_min (ties to paper's definition)
        # n_dims: label-encoded length — used for candidate pool size
        self._n_hps = len(cs.get_hyperparameters())
        probe = sample_config(cs, np.random.RandomState(seed))
        self._n_dims = len(config_to_label_array(probe, cs))
        self._n_min = max(self._N_MIN_FLOOR, self._n_hps + 1)
        self._n_candidates = n_candidates if n_candidates is not None \
            else max(64, 4 * self._n_dims)

        self._surrogate = _RandomForestSurrogate(
            n_estimators=n_estimators,
            min_samples_leaf=min_samples_leaf,
            rng=self._rng,
        )

        # observation store: fidelity → list of (encoded_config, metric)
        self._obs_by_budget: Dict[float, List[Tuple[np.ndarray, float]]] = {}

        self._brackets: List[_BOHBBracket] = []
        self._bracket_idx = 0
        self._bracket_config_map: Dict = {}

        # diagnostics
        self._fit_count = 0
        self._eval_count = 0
        self._model_samples = 0
        self._random_samples = 0
        self._diag: Dict = {"fidelity_history": [], "surrogate_activation_eval": None}
        BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self._diagnostic_dir = os.path.join(BASE_DIR, "results", "diagnostics")
        self._diagnostic_path: Optional[str] = None  # set via set_diagnostic_prefix()

        self._create_brackets()

    # ------------------------------------------------------------------
    # Surrogate management
    # ------------------------------------------------------------------

    def _select_training_budget(self):
        """Return highest fidelity with >= N_min observations, or None."""
        eligible = [b for b, obs in self._obs_by_budget.items() if len(obs) >= self._n_min]
        return max(eligible) if eligible else None

    def _maybe_fit_surrogate(self):
        """Fit RF on observations from the best eligible fidelity. Returns b* or None."""
        b_star = self._select_training_budget()
        if b_star is None:
            return None
        obs = self._obs_by_budget[b_star]
        X = np.array([enc for enc, _ in obs])
        y = np.array([r for _, r in obs])
        self._surrogate.fit(X, y)

        # record diagnostics
        self._fit_count += 1
        n_total = sum(len(v) for v in self._obs_by_budget.values())
        self._diag["fidelity_history"].append({
            "fit_id": self._fit_count,
            "b_star": b_star,
            "n_obs_at_b_star": len(obs),
            "n_total_obs": n_total,
            "model_samples": self._model_samples,
            "random_samples": self._random_samples,
        })
        if self._diag["surrogate_activation_eval"] is None:
            self._diag["surrogate_activation_eval"] = self._eval_count
        self._flush_diagnostics()
        return b_star

    # ------------------------------------------------------------------
    # Configuration sampling
    # ------------------------------------------------------------------

    def _sample_with_model(self, b_star):
        """Sample config with highest EI over a random candidate pool."""
        best_f = max(r for _, r in self._obs_by_budget[b_star])
        candidates = [sample_config(self._cs, self._rng) for _ in range(self._n_candidates)]
        X_cand = np.array([config_to_label_array(c, self._cs) for c in candidates])
        mean, std = self._surrogate.predict(X_cand)
        ei = _expected_improvement(mean, std, best_f, xi=self._xi)
        return candidates[int(np.argmax(ei))]

    def _random_sample(self):
        return sample_config(self._cs, self._rng)

    def _make_config_sampler(self, surrogate_ready, b_star):
        """Return a per-config callable that mixes model-guided and random sampling.

        Fraction rho of configs are always drawn uniformly at random (Algorithm 2,
        line 1 of Falkner et al.), preserving Hyperband's convergence guarantee.
        """
        def sampler():
            if surrogate_ready and self._rng.rand() >= self._rho:
                self._model_samples += 1
                return self._sample_with_model(b_star)
            self._random_samples += 1
            return self._random_sample()
        return sampler

    # ------------------------------------------------------------------
    # Bracket management
    # ------------------------------------------------------------------

    def _create_brackets(self):
        """Instantiate s_max+1 SH brackets and initialise based on refit_mode."""
        self._brackets = []
        self._bracket_config_map = {}
        self._bracket_idx = 0

        for s in range(self._s_max, -1, -1):
            n_s = math.ceil((self._B / self.max_budget) * (self._eta ** s / (s + 1)))
            r_s = max(self.max_budget * self._eta ** (-s), self.min_budget)
            self._brackets.append(
                _BOHBBracket(self._cs, n_s, r_s, self.max_budget, self._eta, self._rng)
            )

        b_star = self._maybe_fit_surrogate()
        sampler = self._make_config_sampler(b_star is not None, b_star)

        if self._refit_mode == "round":
            # all brackets share the same model snapshot
            for bracket in self._brackets:
                bracket.initialise(sampler)
        else:
            # bracket mode: only initialise the first bracket now;
            # remaining brackets are initialised lazily in ask()
            self._brackets[0].initialise(sampler)

    # ------------------------------------------------------------------
    # HPOAlgorithm interface
    # ------------------------------------------------------------------

    def ask(self):
        while True:
            if self._bracket_idx >= len(self._brackets):
                self._create_brackets()
            try:
                config, budget = self._brackets[self._bracket_idx].ask()
                self._bracket_config_map[config_key(config)] = self._bracket_idx
                return config, budget
            except StopIteration:
                self._bracket_idx += 1
                # bracket mode: initialise the next bracket with a fresh surrogate fit
                if self._refit_mode == "bracket" and self._bracket_idx < len(self._brackets):
                    b_star = self._maybe_fit_surrogate()
                    sampler = self._make_config_sampler(b_star is not None, b_star)
                    self._brackets[self._bracket_idx].initialise(sampler)

    def tell(self, config, result, budget):
        # store observation
        encoded = config_to_label_array(config, self._cs)
        if budget not in self._obs_by_budget:
            self._obs_by_budget[budget] = []
        self._obs_by_budget[budget].append((encoded, float(result)))
        self._eval_count += 1

        # forward result to the bracket that issued this config
        key = config_key(config)
        idx = self._bracket_config_map.get(key, self._bracket_idx)
        if 0 <= idx < len(self._brackets):
            self._brackets[idx].tell(config, result, budget)

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def set_diagnostic_prefix(self, prefix):
        """Set the output path for diagnostic JSON (call before running)."""
        os.makedirs(self._diagnostic_dir, exist_ok=True)
        self._diagnostic_path = os.path.join(self._diagnostic_dir, f"{prefix}.json")

    def _flush_diagnostics(self):
        if self._diagnostic_path is None:
            return
        os.makedirs(self._diagnostic_dir, exist_ok=True)
        with open(self._diagnostic_path, "w") as f:
            json.dump(self._diag, f)


class BOHB_Round(BOHB):
    """BOHB with per-round surrogate refitting (all brackets share the same model)."""
    def __init__(self, cs, total_budget, min_budget, max_budget, seed=0):
        super().__init__(cs, total_budget, min_budget, max_budget,
                         seed=seed, refit_mode="round")


class BOHB_Bracket(BOHB):
    """BOHB with per-bracket surrogate refitting (each bracket gets a fresh fit)."""
    def __init__(self, cs, total_budget, min_budget, max_budget, seed=0):
        super().__init__(cs, total_budget, min_budget, max_budget,
                         seed=seed, refit_mode="bracket")
