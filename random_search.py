import numpy as np
from ConfigSpace import Configuration, ConfigurationSpace
from hpo_algorithm import HPOAlgorithm
from utils import sample_config


class RandomSearch(HPOAlgorithm):
    def __init__(self, cs, total_budget, min_budget, max_budget, seed=0):
        super().__init__(cs, total_budget, min_budget, max_budget)
        self._rng = np.random.RandomState(seed)

    def ask(self) -> tuple[Configuration, float]:
        return sample_config(self.cs, self._rng), self.max_budget

    def tell(self, config, result, budget):
        pass
