# HPO Assignment 1 — Full Implementation Plan (v2)
**RWTH Aachen University | Hyperparameter Optimisation for Machine Learning**
**Deadline: 5 May 2026, 23:59 CEST**
**Revised to address all critical and minor issues found in v1 review.**

---

## Overview

We implement five HPO algorithms (Random Search, Grid Search, Successive Halving, Hyperband, Bayesian Optimisation), benchmark them on two YAHPO Gym scenarios, visualise results using matplotlib + DeepCAVE (optional), and write a 2-page LaTeX report.

---

## Key Sources & References

- **Hyperband paper**: Li et al., JMLR 2018 — https://jmlr.org/papers/v18/16-558.html
- **YAHPO Gym paper**: Pfisterer et al., AutoML 2022 — https://proceedings.mlr.press/v188/pfisterer22a/pfisterer22a.pdf
- **YAHPO Gym docs**: https://slds-lmu.github.io/yahpo_gym/
- **DeepCAVE paper & docs**: https://jmlr.org/papers/v26/24-1353.html | https://automl.github.io/DeepCAVE/main/
- **ConfigSpace docs**: https://automl.github.io/ConfigSpace/latest/
- **fANOVA API in DeepCAVE**: https://automl.github.io/DeepCAVE/main/api/deepcave.evaluators.fanova.html
- **HpBandSter (reference implementation of HB)**: https://github.com/automl/HpBandSter

---

## Phase 0 — Prerequisites & Critical Constraints

### 🔴 Grid Search Ambiguity (Assignment PDF Contradiction)
Page 1 of the assignment mentions `GRID_SERACH.PY` as a provided file and lists "random search, grid search and Bayesian optimisation" as the "first part." Page 2's "Provided files" list omits it entirely. The project directory also has no `grid_search.py`.

**Decision**: Implement Grid Search anyway. If it is required, we are covered. If not, it becomes a bonus result and goes in the appendix — which only strengthens the report. The `experiment.py` template imports `GridSearch`, which further suggests it is expected.

**TODO**: Clarify with supervisor at next opportunity, but do not block implementation on this.

### 🔴 Optimization Direction — Maximize Throughout
Both benchmarks use accuracy metrics where **higher is better**:
- `nb301` → `val_accuracy` → **maximize**
- `rbv2_xgboost` → `acc` → **maximize**

This affects every algorithm:
- **SH/HB**: "top configs" = **highest result value** (sort descending, keep first `floor(n/eta)`)
- **BO**: EI formula uses `μ(x) - f_best` where `f_best = max(observed_y)` — positive EI when predicted mean exceeds best seen
- **RS/GS**: no learning, direction irrelevant

### 🔴 Forbidden ConfigSpace Methods
The assignment explicitly bans:
- `cs.sample_configuration()` — even the provided `random_search.py` violates this and must be fixed
- `cs.generate_grid()`
- `config.get_array()`

We implement all from scratch using numpy and direct hyperparameter attributes only.

### 🔴 No scipy — numpy-only Throughout
`requirements.txt` lists only: `ConfigSpace`, `scikit-learn`, `numpy`, `yahpo-gym`.
`scipy` is not listed. Although it is installed as a transitive dependency of scikit-learn, directly importing it may be considered a violation by strict graders.

**Decision**: Implement the normal distribution functions with numpy only:
```python
# Standard normal PDF
def _norm_pdf(x):
    return np.exp(-0.5 * x**2) / np.sqrt(2.0 * np.pi)

# Standard normal CDF using error function (numpy built-in)
def _norm_cdf(x):
    return 0.5 * (1.0 + np.erf(x / np.sqrt(2.0)))
```
`np.erf` is part of numpy's ufuncs — no scipy needed.

Also **drop** `scipy.optimize.minimize` for local refinement. Random sampling over N=1000 candidates is sufficient and avoids any package risk.

### Allowed Packages Only
- `ConfigSpace`, `scikit-learn`, `numpy`, `yahpo-gym`

### API Contract (must not be modified)
All algorithms must implement:
- `__init__(cs, total_budget, min_budget, max_budget)` — subclasses may add extra parameters with default values
- `ask() -> (Configuration, float)`
- `tell(config, result, budget) -> None`

---

## Phase 1 — Shared Utility: `utils.py` (New File)

### 1.1 Manual Random Sampler
Replaces the banned `cs.sample_configuration()`.
```python
def sample_config(cs, rng):
    """
    Sample one random Configuration from cs using numpy rng.
    Handles UniformFloat, UniformInteger, Categorical, Ordinal hyperparameters.
    Respects log-scale flags.
    """
    values = {}
    for hp in cs.get_hyperparameters():
        if isinstance(hp, UniformFloatHyperparameter):
            if hp.log:
                values[hp.name] = float(np.exp(rng.uniform(np.log(hp.lower), np.log(hp.upper))))
            else:
                values[hp.name] = float(rng.uniform(hp.lower, hp.upper))

        elif isinstance(hp, UniformIntegerHyperparameter):
            if hp.log:
                # Sample in log space, round to nearest integer, clamp to valid range
                val = int(round(np.exp(rng.uniform(np.log(hp.lower), np.log(hp.upper)))))
                values[hp.name] = int(np.clip(val, hp.lower, hp.upper))
            else:
                values[hp.name] = int(rng.randint(hp.lower, hp.upper + 1))

        elif isinstance(hp, CategoricalHyperparameter):
            idx = rng.randint(0, len(hp.choices))
            values[hp.name] = hp.choices[idx]

        elif isinstance(hp, OrdinalHyperparameter):
            idx = rng.randint(0, len(hp.sequence))
            values[hp.name] = hp.sequence[idx]

    return Configuration(cs, values=values)
```

> **Note on log-space integers**: The `+1` trick used in some implementations is incorrect in log space. We sample from `[log(lower), log(upper)]`, exponentiate, round to the nearest integer, then clamp to `[lower, upper]`. This gives a correct discrete log-uniform distribution.

### 1.2 Config → NumPy Array Encoder
Replaces the banned `config.get_array()`. Used by BO to feed the GP.
```python
def config_to_array(config, cs):
    """
    Encode a Configuration as a fixed-length float numpy array.
    - UniformFloat/Integer: normalized to [0, 1]
    - Categorical: one-hot encoded
    - Ordinal: index normalized to [0, 1]
    Hyperparameters are sorted by name for consistent ordering.
    """
    vec = []
    for hp in sorted(cs.get_hyperparameters(), key=lambda h: h.name):
        val = config[hp.name]
        if isinstance(hp, (UniformFloatHyperparameter, UniformIntegerHyperparameter)):
            normalized = (val - hp.lower) / (hp.upper - hp.lower)
            vec.append(float(normalized))
        elif isinstance(hp, CategoricalHyperparameter):
            one_hot = [1.0 if val == c else 0.0 for c in hp.choices]
            vec.extend(one_hot)
        elif isinstance(hp, OrdinalHyperparameter):
            idx = hp.sequence.index(val)
            vec.append(idx / (len(hp.sequence) - 1) if len(hp.sequence) > 1 else 0.0)
    return np.array(vec, dtype=float)
```

### 1.3 Grid Builder
Replaces the banned `cs.generate_grid()`.
```python
def build_grid(cs, n_points_continuous=5):
    """
    Build a list of all grid Configurations.
    - Continuous/Integer: n_points_continuous evenly spaced values
    - Categorical/Ordinal: all choices
    Uses itertools.product to enumerate all combinations.
    """
    grids = []
    hp_names = []
    for hp in sorted(cs.get_hyperparameters(), key=lambda h: h.name):
        hp_names.append(hp.name)
        if isinstance(hp, UniformFloatHyperparameter):
            grids.append(list(np.linspace(hp.lower, hp.upper, n_points_continuous)))
        elif isinstance(hp, UniformIntegerHyperparameter):
            n = min(n_points_continuous, hp.upper - hp.lower + 1)
            grids.append([int(round(v)) for v in np.linspace(hp.lower, hp.upper, n)])
        elif isinstance(hp, CategoricalHyperparameter):
            grids.append(list(hp.choices))
        elif isinstance(hp, OrdinalHyperparameter):
            grids.append(list(hp.sequence))

    configs = []
    for combo in itertools.product(*grids):
        values = dict(zip(hp_names, combo))
        try:
            configs.append(Configuration(cs, values=values))
        except Exception:
            pass  # skip invalid combos (e.g., violated conditions)
    return configs
```

### 1.4 Config Identity Helper
`Configuration` objects in ConfigSpace do not have a `config_id` attribute. We use `str(config)` or `frozenset(config.items())` as a stable hashable key.
```python
def config_key(config):
    """Return a hashable key for a Configuration (for use as dict key)."""
    return frozenset(config.items())
```

### 1.5 numpy-only Normal Distribution Functions
```python
def _norm_pdf(x):
    """Standard normal PDF — numpy only, no scipy."""
    return np.exp(-0.5 * x**2) / np.sqrt(2.0 * np.pi)

def _norm_cdf(x):
    """Standard normal CDF using np.erf — numpy only, no scipy."""
    return 0.5 * (1.0 + np.erf(x / np.sqrt(2.0)))
```

---

## Phase 2 — Algorithm Implementations

### 2.1 Random Search (`random_search.py`) — Fix

**Only change**: Replace `cs.sample_configuration()` with `sample_config(cs, self._rng)`.

```python
class RandomSearch(HPOAlgorithm):
    def __init__(self, cs, total_budget, min_budget, max_budget, seed=0):
        super().__init__(cs, total_budget, min_budget, max_budget)
        self._rng = np.random.RandomState(seed)

    def ask(self):
        # Sample a random config using manual sampler (sample_configuration is banned)
        return sample_config(self.cs, self._rng), self.max_budget

    def tell(self, config, result, budget):
        pass  # Random Search does not learn from observations
```

---

### 2.2 Grid Search (`grid_search.py`) — New File

**Concept**: Exhaustively enumerate a finite grid over the configuration space and evaluate each point exactly once. When the grid is exhausted, cycle back to the beginning.

**Design decision to document in report**: Grid size per continuous dimension is set to `n_points=5`. This gives `5^d` configs for d-dimensional continuous spaces. For the nb301 benchmark (34D categorical), we enumerate all choices, not a coarser grid. Grid search is the only algorithm with a predetermined evaluation schedule — it does not adapt.

```python
class GridSearch(HPOAlgorithm):
    def __init__(self, cs, total_budget, min_budget, max_budget, n_points=5, seed=0):
        super().__init__(cs, total_budget, min_budget, max_budget)
        # Build the full grid at init time (no generate_grid allowed)
        self._grid = build_grid(cs, n_points_continuous=n_points)
        self._idx = 0  # pointer into the grid

    def ask(self):
        # Return next grid point; cycle if grid is exhausted
        config = self._grid[self._idx % len(self._grid)]
        self._idx += 1
        return config, self.max_budget

    def tell(self, config, result, budget):
        pass  # Grid Search does not learn from observations
```

---

### 2.3 Successive Halving (`successive_halving.py`)

**Paper reference**: Jamieson & Talwalkar (2016). Also the inner loop of Hyperband.

**🔴 Optimization direction**: "top configs" = **highest result value** (maximize accuracy). Sort descending.

#### Algorithm (from Li et al. JMLR 2018 notation)
```
Input: n (num configs), r (min budget per config), η (halving ratio), R (max budget)
T = {n randomly sampled configs}
For rung i = 0, 1, ..., floor(log_η(n)):
    n_i = floor(n * η^(-i))      ← configs in this rung
    r_i = r * η^i                ← budget for this rung  (r_i <= R)
    Evaluate all configs in T at budget r_i
    T = top floor(n_i / η) configs by result (maximize)
```

#### Budget-aware n calculation
To use approximately `total_budget` fidelity units in one full SH run:
```
s_max = floor(log_η(R / r))         where r = min_budget, R = max_budget
n     = ceil((η^(s_max+1)) / (s_max+1))   configs to start with
```

#### State Machine Design

```
State:
  _pending   : deque of (config, budget) ready to be returned by ask()
  _results   : dict {config_key: result} for current rung
  _rungs     : list of (budget, [config, ...]) for each rung
  _rung_idx  : int — which rung we are currently collecting results for
  _waiting   : set of config_keys submitted but not yet tell()-ed
```

**`__init__`**:
1. Compute `s_max`, rung budgets `[r, r*η, r*η², ..., R]`.
2. Sample `n` initial configs with manual sampler.
3. Fill `_pending` with all `n` configs at `min_budget`.

**`ask()`**:
```python
if not self._pending:
    # All rungs exhausted — restart with new random configs or return last best
    # (assignment does not specify restart behaviour; we restart)
    self._restart()
return self._pending.popleft()
```

**`tell(config, result, budget)`**:
```python
key = config_key(config)
self._results[key] = result
self._waiting.discard(key)

if not self._waiting:  # current rung complete
    current_n = len(self._results)
    n_promote = max(1, floor(current_n / self._eta))
    # Sort by result DESCENDING (maximize accuracy)
    top_configs = sorted(self._results_configs, key=lambda c: self._results[config_key(c)], reverse=True)
    promoted = top_configs[:n_promote]
    
    self._rung_idx += 1
    if self._rung_idx < len(self._rungs):
        next_budget = self._rungs[self._rung_idx]
        for cfg in promoted:
            self._pending.append((cfg, next_budget))
        # Reset for next rung
        self._results = {}
        self._waiting = {config_key(c) for c in promoted}
```

---

### 2.4 Hyperband (`hyperband.py`)

**Paper reference**: Li et al., JMLR 2018.

#### Bracket Formula (exact, from the paper)
```
s_max = floor(log_η(R))           where R = max_budget
B     = (s_max + 1) * R           total budget per "round"

For bracket s = s_max, s_max-1, ..., 0:
    n_s = ceil((B / R) * (η^s / (s+1)))   configs in bracket s
    r_s = R * η^(-s)                        starting budget for bracket s
    Run SuccessiveHalving(n_s configs, r_s starting budget, η)
```

Bracket `s = s_max` → most exploratory (many cheap evals).
Bracket `s = 0` → equivalent to random search at full budget.

#### 🔴 `experiment.py` Fix: Hyperband Must Be in the Optimizer Loop
The provided `experiment.py` is missing `Hyperband` from the optimizer list. This must be added. See Phase 3.

#### State Machine Design

```python
class Hyperband(HPOAlgorithm):
    def __init__(self, cs, total_budget, min_budget, max_budget, eta=3, seed=0):
        super().__init__(cs, total_budget, min_budget, max_budget)
        self._eta = eta
        self._rng = np.random.RandomState(seed)
        
        # Compute brackets
        s_max = floor(log(max_budget / min_budget) / log(eta))
        B = (s_max + 1) * max_budget
        
        # Create one SH sub-instance per bracket
        self._brackets = []
        self._bracket_config_map = {}  # config_key -> bracket_idx
        
        for s in range(s_max, -1, -1):
            n_s = ceil((B / max_budget) * (eta**s / (s + 1)))
            r_s = max_budget * eta**(-s)
            sh = SuccessiveHalvingBracket(cs, n_s, r_s, max_budget, eta, rng=self._rng)
            self._brackets.append(sh)
        
        self._bracket_idx = 0  # which bracket we are currently serving

    def ask(self):
        # Try current bracket first; advance if exhausted
        while self._bracket_idx < len(self._brackets):
            try:
                config, budget = self._brackets[self._bracket_idx].ask()
                self._bracket_config_map[config_key(config)] = self._bracket_idx
                return config, budget
            except StopIteration:
                self._bracket_idx += 1
        # All brackets done: restart from first bracket
        self._bracket_idx = 0
        return self.ask()

    def tell(self, config, result, budget):
        key = config_key(config)
        bracket_idx = self._bracket_config_map.get(key, 0)
        self._brackets[bracket_idx].tell(config, result, budget)
```

> **Note**: `SuccessiveHalvingBracket` is a refactored internal version of `SuccessiveHalving` that takes `n` and `r` directly (instead of computing them from `total_budget`). The public `SuccessiveHalving` class keeps the standard `__init__` API.

---

### 2.5 Bayesian Optimisation (`bayesian_optimisation.py`)

**Paper reference**: Snoek et al. (2012), "Practical Bayesian Optimization of Machine Learning Algorithms."

#### Algorithm
```
Init: collect n_initial random observations (default: 5)
Each ask():
    if not enough observations: return random config
    else:
        1. Encode all observed configs as numpy arrays → X (n × d)
        2. Fit GaussianProcessRegressor to (X, y)
        3. Sample N=1000 random candidate configs, encode to X_cand
        4. For each candidate, compute EI using GP posterior
        5. Return config with highest EI
```

#### 🔴 Surrogate: sklearn GaussianProcessRegressor (mandatory per assignment)
```python
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel

kernel = Matern(nu=2.5) + WhiteKernel(noise_level=1e-6)
gp = GaussianProcessRegressor(
    kernel=kernel,
    normalize_y=True,          # normalize outputs — important for numerical stability
    n_restarts_optimizer=5,    # multiple random restarts for kernel hyperparameter fitting
    random_state=seed
)
```
**Kernel justification for report**: Matérn-5/2 is the standard choice for HPO (used in SMAC, BOHB, scikit-optimize). It assumes functions are twice-differentiable, which is reasonable for smooth ML training curves. `WhiteKernel` accounts for observation noise. `normalize_y=True` handles the scale differences between benchmarks.

#### 🔴 Expected Improvement — Implemented from Scratch (numpy only, no scipy)
```python
def _expected_improvement(X_candidates, gp, y_best, xi=0.01):
    """
    Compute Expected Improvement for a set of candidate points.
    
    Args:
        X_candidates: (N, d) array of encoded candidate configs
        gp:           fitted GaussianProcessRegressor
        y_best:       best observed value so far (maximize)
        xi:           exploration parameter (default 0.01)
    
    Returns:
        ei: (N,) array of EI values
    """
    mu, sigma = gp.predict(X_candidates, return_std=True)
    sigma = np.maximum(sigma, 1e-9)  # avoid division by zero
    
    Z = (mu - y_best - xi) / sigma
    ei = (mu - y_best - xi) * _norm_cdf(Z) + sigma * _norm_pdf(Z)
    ei[sigma < 1e-9] = 0.0  # zero EI where GP is certain
    return ei
```

**Why EI over UCB**: EI is the most widely used acquisition function in HPO. It directly maximises the expected gain over the current best, balancing exploration (high sigma regions) and exploitation (high mean regions). The `xi` parameter controls this trade-off.

**Acquisition optimisation**: Random sampling over N=1000 candidates. No scipy.optimize needed — random sampling is sufficient given the relatively low-dimensional search spaces in YAHPO benchmarks.

#### Full Implementation Structure
```python
class BayesianOptimisation(HPOAlgorithm):
    def __init__(self, cs, total_budget, min_budget, max_budget, n_initial=5, xi=0.01, seed=0):
        super().__init__(cs, total_budget, min_budget, max_budget)
        self._rng = np.random.RandomState(seed)
        self._n_initial = n_initial
        self._xi = xi
        self._X = []    # list of encoded config arrays
        self._y = []    # list of observed results
        self._configs = []  # list of Configuration objects (for reference)
        kernel = Matern(nu=2.5) + WhiteKernel(noise_level=1e-6)
        self._gp = GaussianProcessRegressor(kernel=kernel, normalize_y=True,
                                             n_restarts_optimizer=5, random_state=seed)

    def ask(self):
        if len(self._y) < self._n_initial:
            # Warm-start: random exploration before fitting GP
            return sample_config(self.cs, self._rng), self.max_budget
        
        # Fit GP to observations
        X = np.array(self._X)
        y = np.array(self._y)
        self._gp.fit(X, y)
        
        # Evaluate EI over N=1000 random candidates
        candidates = [sample_config(self.cs, self._rng) for _ in range(1000)]
        X_cand = np.array([config_to_array(c, self.cs) for c in candidates])
        ei = _expected_improvement(X_cand, self._gp, y_best=np.max(y), xi=self._xi)
        
        best_idx = np.argmax(ei)
        return candidates[best_idx], self.max_budget

    def tell(self, config, result, budget):
        self._X.append(config_to_array(config, self.cs))
        self._y.append(float(result))
        self._configs.append(config)
```

---

## Phase 3 — Complete `experiment.py`

### 🔴 All Fixes Required

**Fix 1 — Full function signature** (replace `...` Ellipsis):
```python
def run(optimiser_class, scenario, instance, fidelity_param, metric,
        total_budget, min_budget, max_budget, seed=0):
```

**Fix 2 — Budget variable shadowing** (rename inner `budget` to `eval_budget`):
```python
config, eval_budget = optimiser.ask()   # was: config, budget = optimiser.ask()
cur_budget += eval_budget               # was: cur_budget += None
```

**Fix 3 — Add Hyperband to the optimizer loop** (it was missing in the template):
```python
for optimiser_class in [RandomSearch, BayesianOptimisation, GridSearch, SuccessiveHalving, Hyperband]:
```

**Fix 4 — YAHPO Gym wiring**:
```python
from yahpo_gym import BenchmarkSet

bench = BenchmarkSet(scenario)
bench.set_instance(instance)
cs = bench.get_opt_space(drop_fidelity_params=True)  # HP space WITHOUT fidelity
optimiser = optimiser_class(cs, total_budget, min_budget, max_budget, seed=seed)
```

**Fix 5 — Actual evaluation call**:
```python
config_dict = dict(config)
fidelity_dict = {fidelity_param: eval_budget}
result_dict = bench.objective_function(config_dict, fidelity=fidelity_dict)[0]
result = result_dict[metric]
```

**Fix 6 — 🔴 Save `dict(config)` not raw `Configuration`** (Configuration may not pickle reliably across versions):
```python
runs.append({
    'config': dict(config),          # dict, not Configuration object
    'result': result,
    'eval_budget': eval_budget,
    'cumulative_budget': cur_budget
})
```

**Fix 7 — Seed loop**:
```python
for seed in range(N_SEEDS):
    np.random.seed(seed)
    runs = run(optimiser_class, scenario, instance, fidelity_param, metric,
               total_budget, min_budget, max_budget, seed=seed)
    fname = f"results/{optimiser_class.__name__}_{scenario}_{instance}_seed{seed}.pkl"
    with open(fname, "wb") as f:
        pickle.dump(runs, f)
```

### 🔴 Determining min_budget and max_budget from YAHPO Gym

Do not hardcode — retrieve programmatically:
```python
bench = BenchmarkSet(scenario)
bench.set_instance(instance)
fidelity_space = bench.get_fidelity_space()   # ConfigSpace for fidelity
fid_hp = fidelity_space.get_hyperparameters_dict()[fidelity_param]
min_budget = fid_hp.lower   # e.g., 1 for nb301 epochs
max_budget = fid_hp.upper   # e.g., 98 for nb301 epochs
```

### Budget Strategy

| Benchmark | Discovered min | Discovered max | Recommended total_budget | Why |
|---|---|---|---|---|
| nb301 (epoch) | from YAHPO | from YAHPO | 50 × max_budget | ~50 full evals for RS/BO; ~10 HB rounds |
| rbv2_xgboost (trainsize) | from YAHPO | from YAHPO | 50 × max_budget | same principle |

> **In the report**: Justify this choice explicitly. State that total budget B is set to 50 × max_budget on each benchmark, giving non-multi-fidelity methods approximately 50 full evaluations, while allowing multi-fidelity methods (SH, HB) to explore substantially more configurations at lower costs.

### N_SEEDS = 10
10 seeds provides statistically meaningful mean ± std. Fewer than 5 is not acceptable. Report the mean as a solid line and ± 1 std as a shaded region.

---

## Phase 4 — Experimental Design

### What to measure
- **Incumbent performance at time t**: best result found using cumulative budget ≤ t.
- Formally: `incumbent(t) = max{result_i : cumulative_budget_i ≤ t}`

### Plotting
```python
# For each algorithm, for each seed:
budgets = [r['cumulative_budget'] for r in runs]
results = [r['result'] for r in runs]
incumbents = np.maximum.accumulate(results)  # maximize: running max

# Interpolate to common x-axis, compute mean ± std across seeds
```

### Comparison is fair because
- Total budget B is the same (in fidelity units) for all algorithms
- Multi-fidelity methods consume budget incrementally — each YAHPO call subtracts `eval_budget` from the total
- Non-multi-fidelity methods (RS, GS, BO) always use `max_budget` per call

This is the standard comparison protocol from the HPO literature (Falkner et al., BOHB 2018).

---

## Phase 5 — Visualisation

### Mandatory (matplotlib)

**Figure 1 — Incumbent curves** (one subplot per benchmark, all algorithms overlaid):
- x-axis: cumulative budget consumed (in fidelity units, log scale if range is large)
- y-axis: incumbent metric value (val_accuracy or acc)
- One coloured line per algorithm (consistent colours throughout the paper)
- Shaded region: mean ± 1 std across seeds
- Legend, axis labels with units, clear title

**Figure 2 — Final performance box plot**:
- Distribution of best found metric at end of budget, per algorithm
- One box/violin per algorithm on each benchmark
- Allows direct comparison of final quality

### Optional but Strongly Recommended (DeepCAVE)

**Figure 3 — Hyperparameter Importance (fANOVA)**:
- Export runs to DeepCAVE format using the Recorder or manual conversion
- Use the `Importances` plugin to run fANOVA analysis
- Shows which hyperparameters matter most for each benchmark
- Directly connects to `part2.pdf` lecture content
- Put in appendix if it doesn't fit in 2 pages; mention in the conclusions

---

## Phase 6 — 2-Page LaTeX Report

**Template**: from Moodle. Do not change font size or margins. Figures can be resized.

### Structure

**Introduction** (~2 sentences):
Hyperparameter optimization (HPO) is the problem of finding the configuration of a machine learning algorithm that maximises performance on a given task. We compare four HPO strategies — Random Search, Grid Search, Successive Halving, Hyperband, and Bayesian Optimisation — on two YAHPO Gym benchmarks.

**Methods** (1-2 sentences each):
- **Random Search**: Samples configurations uniformly at random from the search space [Bergstra & Bengio, 2012]. Serves as baseline; surprisingly competitive in high-dimensional spaces.
- **Grid Search**: Exhaustively evaluates a finite discretisation of the search space. Uses 5 points per continuous dimension; number of evaluations grows exponentially with dimension.
- **Successive Halving** [Jamieson & Talwalkar, 2016]: Multi-fidelity method that starts with n configurations at minimum budget, evaluates all, keeps the top 1/η, and doubles the budget. Focuses resources on promising configurations early.
- **Hyperband** [Li et al., 2018]: Runs multiple brackets of Successive Halving with different (n, r) trade-offs, forming a portfolio from explorative to exploitative. Removes the need to pre-specify the number of configurations.
- **Bayesian Optimisation** [Snoek et al., 2012]: Uses a Gaussian Process (Matérn-5/2 kernel) as a surrogate model and Expected Improvement as the acquisition function. Balances exploration and exploitation; justified Matérn-5/2 choice in text.

**Experimental Setup** (~1 paragraph):
- Two YAHPO Gym benchmarks: nb301/cifar10 (val_accuracy, fidelity=epoch) and rbv2_xgboost/instance16 (acc, fidelity=trainsize)
- Same total budget B = 50 × max_budget for all algorithms
- 10 independent seeds per algorithm per benchmark
- Optimization direction: maximize metric
- eta=3 for SH and HB

**Results**:
- Include Figure 1 (incumbent curves) and Figure 2 (final performance)
- Describe: which algorithm converges fastest, which finds the best final solution, and where multi-fidelity methods have an advantage
- If patterns differ across benchmarks, explain why (e.g., the search space structure of nb301 vs rbv2_xgboost)

**Conclusions** (~2 sentences):
Summarise main findings. State which algorithm is recommended for which scenario type.

**Appendix (if space allows)**:
- Figure 3 (fANOVA importance)
- Grid search results if treated as bonus

---

## File Structure After Implementation

```
assignment/
├── hpo_algorithm.py          # base class (DO NOT MODIFY)
├── utils.py                  # NEW: samplers, encoders, grid builder, EI helpers
├── random_search.py          # FIXED: manual sampler instead of sample_configuration
├── grid_search.py            # NEW: grid enumeration (no generate_grid)
├── successive_halving.py     # IMPLEMENTED: state machine with rung promotion
├── hyperband.py              # IMPLEMENTED: multi-bracket wrapper around SH
├── bayesian_optimisation.py  # IMPLEMENTED: GP + EI from scratch, numpy only
├── experiment.py             # COMPLETED: YAHPO, seeds, budget tracking
├── requirements.txt          # unchanged
├── results/                  # auto-created
│   ├── RandomSearch_nb301_cifar10_seed0.pkl
│   ├── RandomSearch_nb301_cifar10_seed1.pkl
│   └── ...
├── figures/                  # generated plots
│   ├── incumbent_curves.pdf
│   ├── final_performance.pdf
│   └── fanova_importance.pdf (optional)
├── report/
│   └── report.tex
└── PLAN.md                   # this document
```

---

## Implementation Order

| Step | File | Depends On | Complexity |
|---|---|---|---|
| 1 | `utils.py` | — | Low |
| 2 | `random_search.py` fix | utils | Trivial |
| 3 | `grid_search.py` | utils | Low |
| 4 | `successive_halving.py` | utils | Medium |
| 5 | `hyperband.py` | utils + SH internals | Medium |
| 6 | `bayesian_optimisation.py` | utils | High |
| 7 | `experiment.py` | all algorithms + YAHPO | Medium |
| 8 | Run experiments | experiment.py | — |
| 9 | Figures | results | Low |
| 10 | Report | figures | Medium |

---

## Summary of All Fixes from v1 Review

| # | Priority | Status | Fix |
|---|---|---|---|
| 1 | 🔴 | ✅ | Grid Search flagged as potentially ambiguous; implemented as bonus-safe |
| 2 | 🔴 | ✅ | Hyperband explicitly added to `experiment.py` optimizer loop |
| 3 | 🔴 | ✅ | scipy replaced with numpy-only `np.erf`-based CDF/PDF |
| 4 | 🔴 | ✅ | Optimization direction (maximize) stated explicitly throughout |
| 5 | 🟡 | ✅ | Log-space integer sampling fixed: `round(exp(...))` + clamp |
| 6 | 🟡 | ✅ | Config identity uses `frozenset(config.items())` not `config_id` |
| 7 | 🟡 | ✅ | `experiment.py` full function signature defined (no `...` Ellipsis) |
| 8 | 🟡 | ✅ | Results saved as `dict(config)` not raw Configuration objects |
| 9 | 🟡 | ✅ | `seed` parameter added to all subclass `__init__` with default=0 |
| 10 | 🟢 | ✅ | fANOVA/importance added to visualisation and report plan |
| 11 | 🟢 | ✅ | YAHPO fidelity bounds retrieved programmatically via `get_fidelity_space()` |
| 12 | 🟢 | ✅ | scipy.optimize local refinement dropped; random sampling only |

---

*Sources consulted during planning:*
- Li et al., Hyperband JMLR 2018: https://jmlr.org/papers/v18/16-558.html
- YAHPO Gym getting started: https://slds-lmu.github.io/yahpo_gym/getting_started.html
- DeepCAVE documentation: https://automl.github.io/DeepCAVE/main/
- DeepCAVE fANOVA API: https://automl.github.io/DeepCAVE/main/api/deepcave.evaluators.fanova.html
- ConfigSpace hyperparameters reference: https://automl.github.io/ConfigSpace/latest/reference/hyperparameters/
- HpBandSter reference implementation: https://github.com/automl/HpBandSter
- Ritchie Vink — BO algorithm breakdown: https://www.ritchievink.com/blog/2019/08/25/algorithm-breakdown-bayesian-optimization/
