from __future__ import annotations

import math

import numpy as np
from ConfigSpace import Configuration, ConfigurationSpace
from hpo_algorithm import HPOAlgorithm
from successive_halving import _SHBracket
from utils import config_key


class Hyperband(HPOAlgorithm):
    def __init__(self, cs, total_budget, min_budget, max_budget, eta=3, seed=0):
        super().__init__(cs, total_budget, min_budget, max_budget)
        self._eta = eta
        self._rng = np.random.RandomState(seed)
        self._cs = cs

        self._s_max = math.floor(math.log(max_budget / min_budget) / math.log(eta))
        self._B = (self._s_max + 1) * max_budget

        self._brackets = []
        self._bracket_idx = 0
        self._bracket_config_map = {}
        self._create_brackets()

    def _create_brackets(self):
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
        while True:
            if self._bracket_idx >= len(self._brackets):
                self._create_brackets()

            try:
                config, budget = self._brackets[self._bracket_idx].ask()
                self._bracket_config_map[config_key(config)] = self._bracket_idx
                return config, budget
            except StopIteration:
                self._bracket_idx += 1

    def tell(self, config, result, budget):
        key = config_key(config)
        idx = self._bracket_config_map.get(key, self._bracket_idx)
        self._brackets[idx].tell(config, result, budget)
