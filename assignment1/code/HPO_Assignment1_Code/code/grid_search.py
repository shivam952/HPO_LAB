import numpy as np
from ConfigSpace import Configuration, ConfigurationSpace
from hpo_algorithm import HPOAlgorithm
from utils import build_grid


class GridSearch(HPOAlgorithm):
    # Evaluates configurations from a pre-defined grid at maximum budget
    def __init__(self, cs, total_budget, min_budget, max_budget, n_points=5, seed=0):
        super().__init__(cs, total_budget, min_budget, max_budget)
        # build the grid (falls back to random sampling if it's too big)
        self._grid = build_grid(cs, n_points_continuous=n_points)
        # shuffle the grid so we don't just evaluate one corner of the space
        rng = np.random.RandomState(seed)
        rng.shuffle(self._grid)
        self._idx = 0

    def ask(self) -> tuple[Configuration, float]:
        # loop around the grid if we run out of configs
        config = self._grid[self._idx % len(self._grid)]
        self._idx += 1
        return config, self.max_budget

    def tell(self, config, result, budget):
        pass
