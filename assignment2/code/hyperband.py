from __future__ import annotations

import math

import numpy as np
from ConfigSpace import Configuration, ConfigurationSpace
from hpo_algorithm import HPOAlgorithm
from successive_halving import _SHBracket
from utils import config_key


class Hyperband(HPOAlgorithm):
    """Hyperband (Li et al., 2018).

    Runs a portfolio of s_max+1 Successive Halving brackets, cycled round-robin.
    Bracket s=0 starts directly at r_max, ensuring at least one bracket always
    evaluates at full fidelity.
    """
    def __init__(self, cs, total_budget, min_budget, max_budget, eta=3, seed=0):
        super().__init__(cs, total_budget, min_budget, max_budget)
        self._eta = eta
        self._rng = np.random.RandomState(seed)
        self._cs = cs

        # s_max: number of promotion rungs in the most exploratory bracket.
        # B: total budget allocated to one full round of all brackets,
        #    set to (s_max+1) * r_max so every bracket receives equal total budget.
        self._s_max = math.floor(math.log(max_budget / min_budget) / math.log(eta))
        self._B = (self._s_max + 1) * max_budget

        self._brackets = []
        self._bracket_idx = 0
        # Maps config_key → bracket index so tell() routes to the correct bracket.
        self._bracket_config_map = {}

        self._create_brackets()

    def _create_brackets(self):
        """Instantiate the full portfolio of s_max+1 SH brackets.

        For each bracket index s (from s_max down to 0):
          n_s = ceil(B/r_max * eta^s / (s+1))  — initial config count
          r_s = max(r_max * eta^(-s), r_min)    — starting fidelity

        s=s_max: many configs, low starting budget (most exploratory).
        s=0:     few configs, starting budget = r_max (most exploitative;
                 this bracket always evaluates at full fidelity).
        """
        self._brackets = []
        self._bracket_config_map = {}
        self._bracket_idx = 0

        for s in range(self._s_max, -1, -1):
            n_s = math.ceil((self._B / self.max_budget) * (self._eta ** s / (s + 1)))
            r_s = max(self.max_budget * self._eta ** (-s), self.min_budget)
            self._brackets.append(
                _SHBracket(self._cs, n_s, r_s, self.max_budget, self._eta, self._rng)
            )

    def ask(self) -> tuple[Configuration, float]:
        """Return the next (config, budget) to evaluate.

        Cycles through brackets in order (s_max → 0). Skips exhausted brackets.
        When all brackets are done, restarts the full portfolio from scratch.
        Returns: (Configuration, float) — config to evaluate and fidelity budget.
        """
        while True:
            if self._bracket_idx >= len(self._brackets):
                # All brackets exhausted — restart the full HB round.
                self._create_brackets()

            try:
                config, budget = self._brackets[self._bracket_idx].ask()
                # Record which bracket this config belongs to so tell() can route it.
                self._bracket_config_map[config_key(config)] = self._bracket_idx
                return config, budget

            except StopIteration:
                # Current bracket exhausted — move to the next one.
                self._bracket_idx += 1

    def tell(self, config, result, budget):
        """Record evaluation result and forward to the bracket that requested it.

        Args:
            config: Configuration that was evaluated.
            result: Observed metric value (higher is better).
            budget: Fidelity budget used for this evaluation.
        """
        key = config_key(config)
        # Look up which bracket issued this config; fall back to current bracket
        # if the key is missing (should not happen in synchronous execution).
        idx = self._bracket_config_map.get(key, self._bracket_idx)
        self._brackets[idx].tell(config, result, budget)
