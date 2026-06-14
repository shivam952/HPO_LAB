import numpy as np
from ConfigSpace import Configuration, ConfigurationSpace
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel

from hpo_algorithm import HPOAlgorithm
from utils import sample_config, config_to_array, _norm_pdf, _norm_cdf


class BayesianOptimisation(HPOAlgorithm):
    # BO with a GP surrogate and Expected Improvement, evaluating at max budget
    def __init__(self, cs, total_budget, min_budget, max_budget,
                 n_initial=5, xi=0.01, seed=0):
        super().__init__(cs, total_budget, min_budget, max_budget)
        self._rng = np.random.RandomState(seed)
        
        # number of random warm-starts before fitting GP
        self._n_initial = n_initial
        # exploration-exploitation trade-off
        self._xi = xi

        self._X = []
        self._y = []

        # Matern-5/2 is standard for HPO; WhiteKernel handles observation noise
        kernel = Matern(nu=2.5) + WhiteKernel(noise_level=1e-6)
        self._gp = GaussianProcessRegressor(
            kernel=kernel,
            normalize_y=True,
            n_restarts_optimizer=5,
            random_state=seed,
        )

    def ask(self) -> tuple[Configuration, float]:
        # warm-start: return random configs until we have enough data
        if len(self._y) < self._n_initial:
            return sample_config(self.cs, self._rng), self.max_budget

        # fit the GP surrogate
        X = np.array(self._X)
        y = np.array(self._y)
        self._gp.fit(X, y)

        # sample 1000 random candidates to optimize EI
        candidates = [sample_config(self.cs, self._rng) for _ in range(1000)]
        X_cand = np.array([config_to_array(c, self.cs) for c in candidates])
        
        ei = self._expected_improvement(X_cand, y_best=np.max(y))

        # return candidate with max EI
        return candidates[np.argmax(ei)], self.max_budget

    def tell(self, config, result, budget):
        self._X.append(config_to_array(config, self.cs))
        self._y.append(float(result))

    def _expected_improvement(self, X_candidates, y_best):
        mu, sigma = self._gp.predict(X_candidates, return_std=True)
        # prevent div by zero
        sigma = np.maximum(sigma, 1e-9)

        # EI(x) = (mu - f_best - xi) * Phi(Z) + sigma * phi(Z)
        improvement = mu - y_best - self._xi
        Z = improvement / sigma
        
        ei = improvement * _norm_cdf(Z) + sigma * _norm_pdf(Z)
        ei[sigma < 1e-9] = 0.0
        return ei
