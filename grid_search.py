import numpy as np
from ConfigSpace import Configuration, ConfigurationSpace
from hpo_algorithm import HPOAlgorithm
from utils import build_grid


class GridSearch(HPOAlgorithm):
    def __init__(self, cs, total_budget, min_budget, max_budget, n_points=5, seed=0):
        super().__init__(cs, total_budget, min_budget, max_budget)
        self._grid = build_grid(cs, n_points_continuous=n_points)
        rng = np.random.RandomState(seed)
        rng.shuffle(self._grid)
        self._idx = 0

    def ask(self) -> tuple[Configuration, float]:
        config = self._grid[self._idx % len(self._grid)]
        self._idx += 1
        return config, self.max_budget

    def tell(self, config, result, budget):
        pass
