"""
Shared utility functions for HPO assignment.

Provides manual replacements for banned ConfigSpace methods:
  - sample_config()    replaces cs.sample_configuration()
  - config_to_array()  replaces config.get_array()
  - build_grid()       replaces cs.generate_grid()

Also provides:
  - config_key()       stable hashable identity for Configuration objects
  - _norm_pdf/_norm_cdf  numpy-only normal distribution (no scipy)
"""

import itertools
import numpy as np
from ConfigSpace import Configuration, ConfigurationSpace
from ConfigSpace.hyperparameters import (
    UniformFloatHyperparameter,
    UniformIntegerHyperparameter,
    CategoricalHyperparameter,
    OrdinalHyperparameter,
    Constant,
)


# ---------------------------------------------------------------------------
# 1. Manual Random Sampler
# ---------------------------------------------------------------------------

def _sample_hp_value(hp, rng: np.random.RandomState):
    """
    Sample a single value for one hyperparameter using *rng*.

    Supports: UniformFloat, UniformInteger, Categorical, Ordinal, Constant.
    Respects the ``log`` flag on continuous / integer parameters.
    """
    if isinstance(hp, Constant):
        return hp.value

    if isinstance(hp, UniformFloatHyperparameter):
        if hp.log:
            return float(np.exp(rng.uniform(np.log(hp.lower), np.log(hp.upper))))
        return float(rng.uniform(hp.lower, hp.upper))

    if isinstance(hp, UniformIntegerHyperparameter):
        if hp.log:
            raw = np.exp(rng.uniform(np.log(hp.lower), np.log(hp.upper)))
            val = int(round(raw))
            return int(np.clip(val, hp.lower, hp.upper))
        return int(rng.randint(hp.lower, hp.upper + 1))

    if isinstance(hp, CategoricalHyperparameter):
        return hp.choices[rng.randint(0, len(hp.choices))]

    if isinstance(hp, OrdinalHyperparameter):
        return hp.sequence[rng.randint(0, len(hp.sequence))]

    raise TypeError(f"Unsupported hyperparameter type: {type(hp)}")


def sample_config(cs: ConfigurationSpace, rng: np.random.RandomState) -> Configuration:
    """
    Sample one random Configuration from ``cs`` using numpy RNG.

    Correctly handles **conditional configuration spaces** (e.g. nb301
    with 24 conditions, rbv2_xgboost with 8 conditions + a Constant HP).

    Strategy:
      1. Sample a value for every HP (including potentially inactive ones).
      2. Create a temporary Configuration that tolerates inactive values.
      3. Query which HPs are actually active given the sampled parents.
      4. Rebuild a clean Configuration with only active HP values.

    Parameters
    ----------
    cs : ConfigurationSpace
        The configuration space to sample from (may have conditions).
    rng : np.random.RandomState
        Seeded numpy random state for reproducibility.

    Returns
    -------
    Configuration
        A valid, randomly sampled configuration (respects conditions).
    """
    # Step 1: sample a raw value for every hyperparameter
    all_values = {}
    for hp in cs.get_hyperparameters():
        all_values[hp.name] = _sample_hp_value(hp, rng)

    # Step 2+3+4: if the space has conditions, prune inactive HPs
    conditions = cs.get_conditions()
    if not conditions:
        # No conditions — all HPs are always active
        return Configuration(cs, values=all_values)

    # Build a temporary config that allows setting inactive HPs, then
    # use ConfigSpace's own logic to determine which are active.
    try:
        tmp = Configuration(cs, values=all_values,
                            allow_inactive_with_values=True)
    except TypeError:
        # Older ConfigSpace versions: fall back to rejection sampling
        for _ in range(1000):
            try:
                return Configuration(cs, values=all_values)
            except ValueError:
                # Re-sample everything and try again
                for hp in cs.get_hyperparameters():
                    all_values[hp.name] = _sample_hp_value(hp, rng)
        raise RuntimeError("Could not sample a valid config after 1000 tries")

    # Keep only active HP values
    active_values = {
        name: all_values[name]
        for name in cs.get_active_hyperparameters(tmp)
    }
    return Configuration(cs, values=active_values)



# ---------------------------------------------------------------------------
# 2. Config → NumPy Array Encoder (for BO's GP)
# ---------------------------------------------------------------------------

def config_to_array(config: Configuration, cs: ConfigurationSpace) -> np.ndarray:
    """
    Encode a Configuration as a fixed-length float numpy array.

    Encoding scheme (hyperparameters sorted by name for consistency):
      - UniformFloat / UniformInteger → normalised to [0, 1]
      - Categorical → one-hot vector of length ``len(choices)``
      - Ordinal → index normalised to [0, 1]

    Parameters
    ----------
    config : Configuration
        The configuration to encode.
    cs : ConfigurationSpace
        The space ``config`` belongs to (supplies bounds and structure).

    Returns
    -------
    np.ndarray
        1-D float array suitable for GP fitting.
    """
    vec = []
    config_dict = dict(config)
    for hp in sorted(cs.get_hyperparameters(), key=lambda h: h.name):
        if isinstance(hp, Constant):
            # Constants carry no information — skip or encode as 0
            continue

        if hp.name not in config_dict:
            # Inactive HP (conditional, parent condition not met)
            # Use a default encoding: 0.0 for numeric, all-zeros for one-hot
            if isinstance(hp, (UniformFloatHyperparameter, UniformIntegerHyperparameter)):
                vec.append(0.0)
            elif isinstance(hp, CategoricalHyperparameter):
                vec.extend([0.0] * len(hp.choices))
            elif isinstance(hp, OrdinalHyperparameter):
                vec.append(0.0)
            continue

        val = config_dict[hp.name]
        if isinstance(hp, (UniformFloatHyperparameter, UniformIntegerHyperparameter)):
            # Linear normalisation to [0, 1]
            denom = hp.upper - hp.lower
            normalised = (val - hp.lower) / denom if denom > 0 else 0.0
            vec.append(float(normalised))
        elif isinstance(hp, CategoricalHyperparameter):
            # One-hot encoding
            one_hot = [1.0 if val == c else 0.0 for c in hp.choices]
            vec.extend(one_hot)
        elif isinstance(hp, OrdinalHyperparameter):
            # Index normalised to [0, 1]
            idx = list(hp.sequence).index(val)
            normalised = idx / (len(hp.sequence) - 1) if len(hp.sequence) > 1 else 0.0
            vec.append(float(normalised))
    return np.array(vec, dtype=np.float64)


# ---------------------------------------------------------------------------
# 3. Grid Builder
# ---------------------------------------------------------------------------

def build_grid(
    cs: ConfigurationSpace,
    n_points_continuous: int = 5,
    max_configs: int = 10_000,
    seed: int = 0,
) -> list[Configuration]:
    """
    Build a list of grid Configurations by enumerating (or sampling from)
    the Cartesian product of per-hyperparameter grids.

    Grid construction (sorted by HP name):
      - UniformFloat → ``n_points_continuous`` linearly-spaced values
        (log-spaced when ``hp.log=True``)
      - UniformInteger → up to ``n_points_continuous`` evenly-spaced integers
      - Categorical → all choices
      - Ordinal → entire sequence

    **Size cap**: high-dimensional spaces (e.g. nb301's 34-D categorical
    space) can yield a Cartesian product too large to enumerate.  When the
    estimated total exceeds ``max_configs``, the function randomly samples
    ``max_configs`` unique combos from the product instead of iterating
    over all of them.  Invalid combinations (violated ConfigSpace conditions)
    are silently skipped in both paths.

    Parameters
    ----------
    cs : ConfigurationSpace
        The configuration space to discretise.
    n_points_continuous : int
        Number of grid points per continuous / integer dimension.
    max_configs : int
        Maximum number of configurations to return.  When the full grid
        exceeds this limit, random sampling is used.
    seed : int
        RNG seed used during random sampling (reproducibility).

    Returns
    -------
    list[Configuration]
        Up to ``max_configs`` valid grid configurations.
    """
    grids = []
    hp_names = []

    for hp in sorted(cs.get_hyperparameters(), key=lambda h: h.name):
        hp_names.append(hp.name)
        if isinstance(hp, UniformFloatHyperparameter):
            if hp.log:
                grids.append(
                    list(np.exp(np.linspace(
                        np.log(hp.lower), np.log(hp.upper), n_points_continuous
                    )))
                )
            else:
                grids.append(list(np.linspace(hp.lower, hp.upper, n_points_continuous)))

        elif isinstance(hp, UniformIntegerHyperparameter):
            n = min(n_points_continuous, hp.upper - hp.lower + 1)
            points = np.linspace(hp.lower, hp.upper, n)
            # Deduplicate after rounding (can happen with small integer ranges)
            grids.append(sorted(set(int(round(v)) for v in points)))

        elif isinstance(hp, CategoricalHyperparameter):
            grids.append(list(hp.choices))

        elif isinstance(hp, OrdinalHyperparameter):
            grids.append(list(hp.sequence))

    # ---- Check for conditional spaces ----
    has_conditions = len(cs.get_conditions()) > 0

    # ---- Estimate total grid size without materialising it ----
    total_size = 1
    for g in grids:
        total_size *= len(g)
        if total_size > max_configs * 100:
            break  # early-exit: definitely need sampling path

    configs: list[Configuration] = []

    if has_conditions:
        # Conditional space (e.g. nb301 with 24 conditions) — Cartesian product
        # combos almost all violate conditions. Use sample_config() which
        # correctly handles conditions via allow_inactive_with_values.
        rng = np.random.RandomState(seed)
        seen: set = set()
        for _ in range(max_configs * 5):
            if len(configs) >= max_configs:
                break
            try:
                cfg = sample_config(cs, rng)
                key = config_key(cfg)
                if key not in seen:
                    seen.add(key)
                    configs.append(cfg)
            except Exception:
                pass

    elif total_size <= max_configs:
        # Small enough to enumerate fully
        for combo in itertools.product(*grids):
            values = dict(zip(hp_names, combo))
            try:
                configs.append(Configuration(cs, values=values))
            except Exception:
                pass  # skip invalid combinations
    else:
        # Too large to enumerate — randomly sample unique combos from the grid.
        rng = np.random.RandomState(seed)
        seen: set = set()
        max_attempts = max_configs * 20
        attempts = 0
        while len(configs) < max_configs and attempts < max_attempts:
            combo = tuple(g[rng.randint(len(g))] for g in grids)
            if combo not in seen:
                seen.add(combo)
                values = dict(zip(hp_names, combo))
                try:
                    configs.append(Configuration(cs, values=values))
                except Exception:
                    pass
            attempts += 1

    return configs


# ---------------------------------------------------------------------------
# 4. Config Identity Helper
# ---------------------------------------------------------------------------

def config_key(config: Configuration):
    """
    Return a hashable key for a Configuration (for use as dict/set key).

    Uses ``frozenset(dict(config).items())``, which is stable regardless
    of insertion order or internal ConfigSpace representation.
    """
    return frozenset(dict(config).items())


# ---------------------------------------------------------------------------
# 5. NumPy-Only Normal Distribution Functions (no scipy)
# ---------------------------------------------------------------------------

def _norm_pdf(x: np.ndarray) -> np.ndarray:
    """Standard normal probability density function — numpy only."""
    return np.exp(-0.5 * x ** 2) / np.sqrt(2.0 * np.pi)


# Vectorised erf at module level — created once, reused on every EI call.
# Uses Python stdlib math.erf — no scipy dependency.
import math as _math
_erf_vec = np.vectorize(_math.erf)


def _norm_cdf(x: np.ndarray) -> np.ndarray:
    """Standard normal CDF via ``math.erf`` (stdlib, no scipy needed)."""
    return 0.5 * (1.0 + _erf_vec(x / np.sqrt(2.0)))
