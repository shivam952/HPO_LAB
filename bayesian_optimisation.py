"""
Bayesian Optimisation — HPO Algorithm.

Uses a Gaussian Process (GP) surrogate model with a Matérn-5/2 kernel
and Expected Improvement (EI) as the acquisition function.

Architecture:
  1. Warm-start with ``n_initial`` random evaluations.
  2. Fit GP to all (X_encoded, y) observations.
  3. Sample N=1000 random candidates, compute EI for each.
  4. Return the candidate with the highest EI.

The GP surrogate is ``sklearn.gaussian_process.GaussianProcessRegressor``
(mandatory per assignment).  The acquisition function and its optimisation
are implemented from scratch using numpy only (no scipy).

Optimization direction: **maximize**.  EI is computed as the expected gain
over the current best observation.

Reference: Snoek et al. (2012), "Practical Bayesian Optimization of
Machine Learning Algorithms."
"""

from __future__ import annotations

import numpy as np
from ConfigSpace import Configuration, ConfigurationSpace
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel

from hpo_algorithm import HPOAlgorithm
from utils import sample_config, config_to_array, _norm_pdf, _norm_cdf


class BayesianOptimisation(HPOAlgorithm):
    """
    Bayesian Optimisation with GP + Expected Improvement.

    Parameters
    ----------
    n_initial : int
        Number of random evaluations before fitting the GP (default 5).
    xi : float
        Exploration parameter for EI (default 0.01).  Higher values
        encourage exploration; lower values encourage exploitation.
    seed : int
        Random seed for reproducibility.
    """

    def __init__(
        self,
        cs: ConfigurationSpace,
        total_budget: int,
        min_budget: float,
        max_budget: float,
        n_initial: int = 5,
        xi: float = 0.01,
        seed: int = 0,
    ) -> None:
        super().__init__(cs, total_budget, min_budget, max_budget)
        self._rng = np.random.RandomState(seed)
        self._n_initial = n_initial
        self._xi = xi

        # Observation history
        self._X: list[np.ndarray] = []   # encoded config arrays
        self._y: list[float] = []        # observed results

        # GP surrogate — Matérn-5/2 is the standard kernel for HPO
        # WhiteKernel accounts for observation noise
        kernel = Matern(nu=2.5) + WhiteKernel(noise_level=1e-6)
        self._gp = GaussianProcessRegressor(
            kernel=kernel,
            normalize_y=True,           # important for numerical stability
            n_restarts_optimizer=5,     # multiple restarts for kernel fitting
            random_state=seed,
        )

    def ask(self) -> tuple[Configuration, float]:
        """
        Return the next configuration to evaluate.

        Returns a random config during warm-start (first ``n_initial``
        calls), then uses the GP + EI strategy.
        """
        if len(self._y) < self._n_initial:
            # Warm-start phase: random exploration
            return sample_config(self.cs, self._rng), self.max_budget

        # Fit GP to all observations
        X = np.array(self._X)
        y = np.array(self._y)
        self._gp.fit(X, y)

        # Generate N=1000 random candidate configs
        candidates = [sample_config(self.cs, self._rng) for _ in range(1000)]
        X_cand = np.array([config_to_array(c, self.cs) for c in candidates])

        # Evaluate Expected Improvement for each candidate
        ei = self._expected_improvement(X_cand, y_best=np.max(y))

        # Return the candidate with highest EI
        best_idx = np.argmax(ei)
        return candidates[best_idx], self.max_budget

    def tell(self, config: Configuration, result: float, budget: float) -> None:
        """
        Record a new (config, result) observation for future GP fitting.
        """
        self._X.append(config_to_array(config, self.cs))
        self._y.append(float(result))

    # ----- Private: acquisition function -----

    def _expected_improvement(
        self, X_candidates: np.ndarray, y_best: float
    ) -> np.ndarray:
        """
        Compute Expected Improvement for a batch of candidate points.

        EI(x) = (μ(x) - f_best - ξ) · Φ(Z) + σ(x) · φ(Z)
        where Z = (μ(x) - f_best - ξ) / σ(x)

        Parameters
        ----------
        X_candidates : np.ndarray, shape (N, d)
            Encoded candidate configurations.
        y_best : float
            Best observed value so far (maximize).

        Returns
        -------
        np.ndarray, shape (N,)
            EI values for each candidate.
        """
        mu, sigma = self._gp.predict(X_candidates, return_std=True)
        sigma = np.maximum(sigma, 1e-9)  # avoid division by zero

        improvement = mu - y_best - self._xi
        Z = improvement / sigma

        ei = improvement * _norm_cdf(Z) + sigma * _norm_pdf(Z)
        # Zero out EI where the GP is essentially certain (no exploration value)
        ei[sigma < 1e-9] = 0.0

        return ei
