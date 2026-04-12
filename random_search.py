"""
Random Search — HPO Algorithm.

Samples configurations uniformly at random from the search space and evaluates
each at max_budget.  Does not learn from past observations (tell is a no-op).

This is the simplest possible baseline; it is surprisingly competitive in
high-dimensional spaces (Bergstra & Bengio, 2012).

NOTE: The provided skeleton used ``cs.sample_configuration()``, which is
banned.  This version uses our manual ``sample_config()`` from utils.py.
"""

import numpy as np
from ConfigSpace import Configuration, ConfigurationSpace
from hpo_algorithm import HPOAlgorithm
from utils import sample_config


class RandomSearch(HPOAlgorithm):
    def __init__(
        self,
        cs: ConfigurationSpace,
        total_budget: int,
        min_budget: float,
        max_budget: float,
        seed: int = 0,
    ) -> None:
        """
        Parameters
        ----------
        cs : ConfigurationSpace
            Search space (fidelity parameters already dropped).
        total_budget : int
            Total fidelity budget for the entire optimisation run.
        min_budget, max_budget : float
            Fidelity bounds (only max_budget is used by RS).
        seed : int
            Random seed for reproducibility.
        """
        super().__init__(cs, total_budget, min_budget, max_budget)
        self._rng = np.random.RandomState(seed)

    def ask(self) -> tuple[Configuration, float]:
        """Return a uniformly random config, evaluated at full budget."""
        return sample_config(self.cs, self._rng), self.max_budget

    def tell(self, config: Configuration, result: float, budget: float) -> None:
        """No-op — Random Search does not learn from observations."""
        pass