import numpy as np
from ConfigSpace import Configuration, ConfigurationSpace
from hpo_algorithm import HPOAlgorithm
from utils import sample_config


class RandomSearch(HPOAlgorithm):
    # Evaluates completely random configurations at the maximum budget
    def __init__(self, cs, total_budget, min_budget, max_budget, seed=0):
        super().__init__(cs, total_budget, min_budget, max_budget)
        # local random state for reproducibility
        self._rng = np.random.RandomState(seed)

    def ask(self) -> tuple[Configuration, float]:
        # returns a random config and the max budget
        return sample_config(self.cs, self._rng), self.max_budget

    def tell(self, config, result, budget):
        # RS has no memory so we don't need to track results
        pass
