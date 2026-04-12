"""
Hyperband — HPO Algorithm.

Runs multiple brackets of Successive Halving with different exploration /
exploitation trade-offs (n, r).  This creates a portfolio that is robust
to the unknown optimal early-stopping budget.

Reference: Li et al., JMLR 2018 — "Hyperband: A Novel Bandit-Based
Approach to Hyperparameter Optimization."

Bracket formula:
    s_max = floor(log_η(max_budget / min_budget))
    B     = (s_max + 1) * max_budget

    For s = s_max, s_max-1, ..., 0:
        n_s = ceil((B / max_budget) * (η^s / (s+1)))   # configs
        r_s = max_budget * η^(-s)                       # start budget
        Run SuccessiveHalving(n_s, r_s)

Bracket s=s_max is most exploratory (many cheap evals).
Bracket s=0 is most exploitative (few configs at near-full budget).

Optimization direction: **maximize** (inherited from _SHBracket).
"""

from __future__ import annotations

import math

import numpy as np
from ConfigSpace import Configuration, ConfigurationSpace
from hpo_algorithm import HPOAlgorithm
from successive_halving import _SHBracket
from utils import config_key


class Hyperband(HPOAlgorithm):
    """
    Hyperband optimiser.

    Parameters
    ----------
    eta : int
        Halving ratio shared across all brackets (default 3).
    seed : int
        Random seed for reproducibility.
    """

    def __init__(
        self,
        cs: ConfigurationSpace,
        total_budget: int,
        min_budget: float,
        max_budget: float,
        eta: int = 3,
        seed: int = 0,
    ) -> None:
        super().__init__(cs, total_budget, min_budget, max_budget)
        self._eta = eta
        self._rng = np.random.RandomState(seed)
        self._cs = cs

        # Number of rungs / brackets
        self._s_max = math.floor(
            math.log(max_budget / min_budget) / math.log(eta)
        )

        # Total budget per HB round (for the n_s formula)
        self._B = (self._s_max + 1) * max_budget

        # Create the initial set of brackets
        self._brackets: list[_SHBracket] = []
        self._bracket_idx = 0  # which bracket we are currently serving from
        self._bracket_config_map: dict = {}  # config_key → bracket index

        self._create_brackets()

    def _create_brackets(self) -> None:
        """Instantiate one _SHBracket per s value (s_max down to 0)."""
        self._brackets = []
        for s in range(self._s_max, -1, -1):
            n_s = math.ceil(
                (self._B / self.max_budget) * (self._eta ** s / (s + 1))
            )
            r_s = self.max_budget * self._eta ** (-s)
            # Clamp starting budget to at least min_budget
            r_s = max(r_s, self.min_budget)

            bracket = _SHBracket(
                cs=self._cs,
                n_configs=n_s,
                start_budget=r_s,
                max_budget=self.max_budget,
                eta=self._eta,
                rng=self._rng,
            )
            self._brackets.append(bracket)

    def ask(self) -> tuple[Configuration, float]:
        """
        Return the next (config, budget) pair from the current bracket.

        If the current bracket is exhausted, advances to the next.
        When all brackets are done, creates a fresh round of brackets.
        """
        while True:
            if self._bracket_idx >= len(self._brackets):
                # All brackets exhausted — restart a new HB round
                self._bracket_idx = 0
                self._create_brackets()

            try:
                config, budget = self._brackets[self._bracket_idx].ask()
                key = config_key(config)
                self._bracket_config_map[key] = self._bracket_idx
                return config, budget
            except StopIteration:
                # Current bracket exhausted → move to next
                self._bracket_idx += 1

    def tell(self, config: Configuration, result: float, budget: float) -> None:
        """
        Route the result to the correct bracket for rung promotion.
        """
        key = config_key(config)
        bracket_idx = self._bracket_config_map.get(key, self._bracket_idx)
        self._brackets[bracket_idx].tell(config, result, budget)
