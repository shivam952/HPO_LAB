"""
Grid Search — HPO Algorithm.

Exhaustively evaluates a finite discretisation of the configuration space.
Each configuration is evaluated at max_budget (no multi-fidelity).

Grid construction:
  - Continuous / integer dimensions: ``n_points`` evenly spaced values
  - Categorical / ordinal dimensions: all choices
  - Total grid size = product of per-dimension sizes

When the grid is exhausted, the pointer wraps around (cycle).

Design trade-off documented in report: 5 points per continuous dimension
yields 5^d total configs for d-dimensional continuous spaces, which may
be very large.  For benchmarks with purely categorical spaces (e.g. nb301)
the grid is the full combinatorial space of all choices.
"""

import numpy as np
from ConfigSpace import Configuration, ConfigurationSpace
from hpo_algorithm import HPOAlgorithm
from utils import build_grid


class GridSearch(HPOAlgorithm):
    def __init__(
        self,
        cs: ConfigurationSpace,
        total_budget: int,
        min_budget: float,
        max_budget: float,
        n_points: int = 5,
        seed: int = 0,
    ) -> None:
        """
        Parameters
        ----------
        cs : ConfigurationSpace
            Search space (fidelity parameters already dropped).
        total_budget : int
            Total fidelity budget for the entire run.
        min_budget, max_budget : float
            Fidelity bounds (only max_budget is used — no multi-fidelity).
        n_points : int
            Number of grid points per continuous / integer dimension.
        seed : int
            Random seed (used to shuffle the grid for fairness).
        """
        super().__init__(cs, total_budget, min_budget, max_budget)

        # Build the full grid at init time (generate_grid is banned)
        self._grid = build_grid(cs, n_points_continuous=n_points)

        # Shuffle so that grid traversal order doesn't introduce systematic bias
        rng = np.random.RandomState(seed)
        rng.shuffle(self._grid)

        self._idx = 0  # pointer into the grid

    def ask(self) -> tuple[Configuration, float]:
        """
        Return the next grid configuration, evaluated at full budget.
        Wraps around if the grid is exhausted.
        """
        config = self._grid[self._idx % len(self._grid)]
        self._idx += 1
        return config, self.max_budget

    def tell(self, config: Configuration, result: float, budget: float) -> None:
        """No-op — Grid Search does not learn from observations."""
        pass
