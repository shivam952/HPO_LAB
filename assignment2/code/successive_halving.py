from __future__ import annotations

import math
from collections import deque

import numpy as np
from ConfigSpace import Configuration, ConfigurationSpace
from hpo_algorithm import HPOAlgorithm
from utils import sample_config, config_key


class _SHBracket:
    """A single Successive Halving bracket.

    Kept as a standalone class so Hyperband can instantiate multiple brackets
    with different (n_configs, start_budget) pairs and reuse this logic.

    Args:
        cs:           ConfigurationSpace to sample from.
        n_configs:    Number of configurations to evaluate in the first rung.
        start_budget: Fidelity value for the first rung (r_min for SH, r_s for HB).
        max_budget:   Maximum fidelity value (r_max); rung schedule stops here.
        eta:          Halving rate — keep top 1/eta configs and multiply budget by eta.
        rng:          Seeded numpy RandomState for reproducibility.
    """
    def __init__(self, cs, n_configs, start_budget, max_budget, eta, rng):
        self._eta = eta
        self._rng = rng

        # rung budget schedule: start_budget * eta^k, capped at max_budget
        self._rung_budgets = []
        b = start_budget
        while b <= max_budget + 1e-9:
            self._rung_budgets.append(float(min(b, max_budget)))
            b *= eta

        if not self._rung_budgets:
            self._rung_budgets = [max_budget]

        # Sample the initial pool of n_configs configurations uniformly at random.
        configs = [sample_config(cs, rng) for _ in range(n_configs)]

        # _pending: FIFO queue of (config, budget) pairs waiting to be evaluated.
        # Initially all n_configs are assigned to the first (cheapest) rung budget.
        self._pending: deque = deque((c, self._rung_budgets[0]) for c in configs)
        self._rung_idx = 0               # current position in _rung_budgets
        self._rung_configs = list(configs)  # configs surviving into the current rung
        self._results = {}               # config_key → metric for the current rung only
        self._exhausted = False          # True once the final rung has been promoted

    @property
    def exhausted(self):
        # A bracket is done when it has finished its last rung AND the pending
        # queue is empty (i.e. no more evaluations to request).
        return self._exhausted and len(self._pending) == 0

    def ask(self) -> tuple[Configuration, float]:
        # Returns the next (config, budget) pair to evaluate.
        # Raises StopIteration when all rungs are complete — callers handle restart.
        if not self._pending:
            raise StopIteration
        return self._pending.popleft()

    def tell(self, config, result, budget):
        """Record the result of one evaluation and trigger promotion if the rung is done.

        Args:
            config: The Configuration that was evaluated (same object returned by ask).
            result: The metric value observed (higher is better).
            budget: The fidelity budget used (stored for reference but not re-used here).
        """
        # Store result keyed by a hashable config representation.
        self._results[config_key(config)] = result

        # Wait until every config in the current rung has reported back.
        # Only then do we have a fair comparison to decide who gets promoted.
        if len(self._results) < len(self._rung_configs):
            return

        # --- Rung complete: promote the top 1/eta fraction ---
        # floor(n / eta) gives the number of survivors; at least 1 always survives.
        n_promote = max(1, math.floor(len(self._rung_configs) / self._eta))
        # Sort descending by metric (higher = better) and take the top n_promote.
        ranked = sorted(self._rung_configs,
                        key=lambda c: self._results[config_key(c)], reverse=True)
        promoted = ranked[:n_promote]

        self._rung_idx += 1
        if self._rung_idx < len(self._rung_budgets):
            # There is a next rung: enqueue promoted configs at the higher budget.
            next_budget = self._rung_budgets[self._rung_idx]
            self._rung_configs = promoted
            self._results = {}  # reset: only track current rung's results
            for c in promoted:
                self._pending.append((c, next_budget))
        else:
            # No more rungs — this bracket is finished.
            self._exhausted = True


class SuccessiveHalving(HPOAlgorithm):
    """Successive Halving (Jamieson & Talwalkar, 2016).

    Runs a single SH bracket at a time. When a bracket is exhausted it restarts
    with a fresh set of randomly sampled configurations (no memory across brackets).

    Args:
        cs:           ConfigurationSpace defining the hyperparameter search space.
        total_budget: Total fidelity budget across all evaluations.
        min_budget:   Minimum fidelity per evaluation (first rung, r_min).
        max_budget:   Maximum fidelity per evaluation (r_max).
        eta:          Halving/promotion rate (default 3, as in the original paper).
        seed:         Random seed for reproducibility.
    """
    def __init__(self, cs, total_budget, min_budget, max_budget, eta=3, seed=0):
        super().__init__(cs, total_budget, min_budget, max_budget)
        self._eta = eta
        self._rng = np.random.RandomState(seed)
        self._cs = cs

        # s_max = floor(log_eta(r_max / r_min)): number of promotion steps per bracket.
        # n_initial = ceil(eta^(s_max+1) / (s_max+1)): initial pool size that keeps
        # the total budget per bracket approximately equal to (s_max+1) * r_max.
        self._s_max = math.floor(math.log(max_budget / min_budget) / math.log(eta))
        self._n_initial = math.ceil((eta ** (self._s_max + 1)) / (self._s_max + 1))

        self._bracket = self._new_bracket()

    def _new_bracket(self):
        # Constructs a fresh bracket starting from r_min with n_initial configs.
        return _SHBracket(self._cs, self._n_initial, self.min_budget,
                          self.max_budget, self._eta, self._rng)

    def ask(self) -> tuple[Configuration, float]:
        """Return the next (config, budget) to evaluate.

        If the current bracket is exhausted, silently start a new one.
        Returns: (Configuration, float) — config to evaluate and fidelity budget.
        """
        try:
            return self._bracket.ask()
        except StopIteration:
            # Current bracket done: restart with a new random pool (no memory).
            self._bracket = self._new_bracket()
            return self._bracket.ask()

    def tell(self, config, result, budget):
        """Record evaluation result and forward to the active bracket.

        Args:
            config: Configuration that was evaluated.
            result: Observed metric value (higher is better).
            budget: Fidelity budget used for this evaluation.
        """
        self._bracket.tell(config, result, budget)
