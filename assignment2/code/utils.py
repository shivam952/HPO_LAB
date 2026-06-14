import itertools
import math as _math
import numpy as np
from ConfigSpace import Configuration, ConfigurationSpace
from ConfigSpace.hyperparameters import (
    UniformFloatHyperparameter,
    UniformIntegerHyperparameter,
    CategoricalHyperparameter,
    OrdinalHyperparameter,
    Constant,
)


# ---- Random sampler (replaces banned cs.sample_configuration()) ----

def _sample_hp_value(hp, rng):
    if isinstance(hp, Constant):
        return hp.value
    if isinstance(hp, UniformFloatHyperparameter):
        if hp.log:
            return float(np.exp(rng.uniform(np.log(hp.lower), np.log(hp.upper))))
        return float(rng.uniform(hp.lower, hp.upper))
    if isinstance(hp, UniformIntegerHyperparameter):
        if hp.log:
            val = int(round(np.exp(rng.uniform(np.log(hp.lower), np.log(hp.upper)))))
            return int(np.clip(val, hp.lower, hp.upper))
        return int(rng.randint(hp.lower, hp.upper + 1))
    if isinstance(hp, CategoricalHyperparameter):
        return hp.choices[rng.randint(0, len(hp.choices))]
    if isinstance(hp, OrdinalHyperparameter):
        return hp.sequence[rng.randint(0, len(hp.sequence))]
    raise TypeError(f"Unsupported hyperparameter type: {type(hp)}")


def sample_config(cs: ConfigurationSpace, rng: np.random.RandomState) -> Configuration:
    """Sample a random config, handling conditional spaces correctly."""
    all_values = {hp.name: _sample_hp_value(hp, rng) for hp in cs.get_hyperparameters()}

    if not cs.get_conditions():
        return Configuration(cs, values=all_values)

    try:
        tmp = Configuration(cs, values=all_values, allow_inactive_with_values=True)
    except TypeError:
        for _ in range(1000):
            try:
                return Configuration(cs, values=all_values)
            except ValueError:
                all_values = {hp.name: _sample_hp_value(hp, rng) for hp in cs.get_hyperparameters()}
        raise RuntimeError("Could not sample a valid config after 1000 tries")

    active_values = {name: all_values[name] for name in cs.get_active_hyperparameters(tmp)}
    return Configuration(cs, values=active_values)


# ---- Config encoders ----

def config_to_array(config: Configuration, cs: ConfigurationSpace) -> np.ndarray:
    """Encode config as float array with one-hot categoricals. Used by GP surrogate."""
    vec = []
    config_dict = dict(config)

    for hp in sorted(cs.get_hyperparameters(), key=lambda h: h.name):
        if isinstance(hp, Constant):
            continue

        if hp.name not in config_dict:
            if isinstance(hp, CategoricalHyperparameter):
                vec.extend([0.0] * len(hp.choices))
            else:
                vec.append(0.0)
            continue

        val = config_dict[hp.name]
        if isinstance(hp, (UniformFloatHyperparameter, UniformIntegerHyperparameter)):
            if getattr(hp, 'log', False):
                denom = _math.log(hp.upper) - _math.log(hp.lower)
                normalised = (_math.log(float(val)) - _math.log(hp.lower)) / denom if denom > 0 else 0.0
            else:
                denom = hp.upper - hp.lower
                normalised = (float(val) - hp.lower) / denom if denom > 0 else 0.0
            vec.append(float(np.clip(normalised, 0.0, 1.0)))
        elif isinstance(hp, CategoricalHyperparameter):
            vec.extend([1.0 if val == c else 0.0 for c in hp.choices])
        elif isinstance(hp, OrdinalHyperparameter):
            idx = list(hp.sequence).index(val)
            vec.append(idx / (len(hp.sequence) - 1) if len(hp.sequence) > 1 else 0.0)

    return np.array(vec, dtype=np.float64)


def config_to_label_array(config: Configuration, cs: ConfigurationSpace) -> np.ndarray:
    """Encode config as float array with label-encoded categoricals. Used by BOHB's RF.

    Each categorical HP is mapped to a single value: label(v) = index(v) / (n_choices - 1).
    This keeps NB301's encoded dimension at 34 instead of 272 with one-hot, bringing
    N_min from 273 down to 35 and allowing the surrogate to activate early in the run.
    """
    vec = []
    config_dict = dict(config)

    for hp in sorted(cs.get_hyperparameters(), key=lambda h: h.name):
        if isinstance(hp, Constant):
            continue

        if hp.name not in config_dict:
            vec.append(0.0)
            continue

        val = config_dict[hp.name]
        if isinstance(hp, (UniformFloatHyperparameter, UniformIntegerHyperparameter)):
            if getattr(hp, 'log', False):
                denom = _math.log(hp.upper) - _math.log(hp.lower)
                normalised = (_math.log(float(val)) - _math.log(hp.lower)) / denom if denom > 0 else 0.0
            else:
                denom = hp.upper - hp.lower
                normalised = (float(val) - hp.lower) / denom if denom > 0 else 0.0
            vec.append(float(np.clip(normalised, 0.0, 1.0)))
        elif isinstance(hp, CategoricalHyperparameter):
            idx = list(hp.choices).index(val)
            vec.append(idx / (len(hp.choices) - 1) if len(hp.choices) > 1 else 0.0)
        elif isinstance(hp, OrdinalHyperparameter):
            idx = list(hp.sequence).index(val)
            vec.append(idx / (len(hp.sequence) - 1) if len(hp.sequence) > 1 else 0.0)

    return np.array(vec, dtype=np.float64)


# ---- Grid builder (replaces banned cs.generate_grid()) ----

def build_grid(cs: ConfigurationSpace, n_points_continuous=5, max_configs=10_000, seed=0):
    """Build a grid of configs; falls back to random sampling for large spaces."""
    grids, hp_names = [], []

    for hp in sorted(cs.get_hyperparameters(), key=lambda h: h.name):
        hp_names.append(hp.name)
        if isinstance(hp, UniformFloatHyperparameter):
            pts = np.exp(np.linspace(np.log(hp.lower), np.log(hp.upper), n_points_continuous)) \
                  if hp.log else np.linspace(hp.lower, hp.upper, n_points_continuous)
            grids.append(list(pts))
        elif isinstance(hp, UniformIntegerHyperparameter):
            n = min(n_points_continuous, hp.upper - hp.lower + 1)
            grids.append(sorted(set(int(round(v)) for v in np.linspace(hp.lower, hp.upper, n))))
        elif isinstance(hp, CategoricalHyperparameter):
            grids.append(list(hp.choices))
        elif isinstance(hp, OrdinalHyperparameter):
            grids.append(list(hp.sequence))
        elif isinstance(hp, Constant):
            grids.append([hp.value])

    has_conditions = len(cs.get_conditions()) > 0
    total_size = 1
    for g in grids:
        total_size *= len(g)
        if total_size > max_configs * 100:
            break

    configs = []
    rng = np.random.RandomState(seed)

    if has_conditions:
        # Conditional spaces (e.g. NB301): full Cartesian product is invalid.
        # Fall back to random sampling which respects conditions via sample_config().
        seen = set()
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
        for combo in itertools.product(*grids):
            try:
                configs.append(Configuration(cs, values=dict(zip(hp_names, combo))))
            except Exception:
                pass
    else:
        # Space too large: sample random combinations from the discretised grid values.
        seen = set()
        for _ in range(max_configs * 20):
            if len(configs) >= max_configs:
                break
            combo = tuple(g[rng.randint(len(g))] for g in grids)
            if combo not in seen:
                seen.add(combo)
                try:
                    configs.append(Configuration(cs, values=dict(zip(hp_names, combo))))
                except Exception:
                    pass

    return configs


# ---- Helpers ----

def config_key(config: Configuration):
    return frozenset(dict(config).items())


def _norm_pdf(x):
    return np.exp(-0.5 * x ** 2) / np.sqrt(2.0 * np.pi)


_erf_vec = np.vectorize(_math.erf)


def _norm_cdf(x):
    return 0.5 * (1.0 + _erf_vec(x / np.sqrt(2.0)))
