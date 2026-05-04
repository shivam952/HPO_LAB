from __future__ import annotations

import math
from collections import deque

import numpy as np
from ConfigSpace import Configuration, ConfigurationSpace
from hpo_algorithm import HPOAlgorithm
from utils import sample_config, config_key


class _SHBracket:
    """Single SH bracket — reused by Hyperband."""

    def __init__(self, cs, n_configs, start_budget, max_budget, eta, rng):
        self._eta = eta
        self._rng = rng

        # Build rung schedule: start_budget, start_budget*eta, ..., <= max_budget
        self._rung_budgets = []
        b = start_budget
        while b <= max_budget + 1e-9:
            self._rung_budgets.append(float(min(b, max_budget)))
            b *= eta
        if not self._rung_budgets:
            self._rung_budgets = [max_budget]

        configs = [sample_config(cs, rng) for _ in range(n_configs)]
        self._pending: deque = deque((c, self._rung_budgets[0]) for c in configs)
        self._rung_idx = 0
        self._rung_configs = list(configs)
        self._results = {}
        self._exhausted = False

    @property
    def exhausted(self):
        return self._exhausted and len(self._pending) == 0

    def ask(self) -> tuple[Configuration, float]:
        if not self._pending:
            raise StopIteration
        return self._pending.popleft()

    def tell(self, config, result, budget):
        self._results[config_key(config)] = result

        if len(self._results) < len(self._rung_configs):
            return  # rung not complete yet

        # Promote top 1/eta configs to next rung
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


class SuccessiveHalving(HPOAlgorithm):
    def __init__(self, cs, total_budget, min_budget, max_budget, eta=3, seed=0):
        super().__init__(cs, total_budget, min_budget, max_budget)
        self._eta = eta
        self._rng = np.random.RandomState(seed)
        self._cs = cs

        self._s_max = math.floor(math.log(max_budget / min_budget) / math.log(eta))
        self._n_initial = math.ceil((eta ** (self._s_max + 1)) / (self._s_max + 1))
        self._bracket = self._new_bracket()

    def _new_bracket(self):
        return _SHBracket(self._cs, self._n_initial, self.min_budget,
                          self.max_budget, self._eta, self._rng)

    def ask(self) -> tuple[Configuration, float]:
        try:
            return self._bracket.ask()
        except StopIteration:
            self._bracket = self._new_bracket()
            return self._bracket.ask()

    def tell(self, config, result, budget):
        self._bracket.tell(config, result, budget)
