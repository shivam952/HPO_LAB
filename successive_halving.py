"""
Successive Halving — HPO Algorithm.

Multi-fidelity method that evaluates many configurations at a low budget,
keeps the top 1/η fraction, and promotes survivors to the next (higher)
budget rung.  Repeats until only one configuration remains at max_budget.

Reference: Jamieson & Talwalkar (2016); also the inner loop of
Hyperband (Li et al., JMLR 2018).

Optimization direction: **maximize** (higher result = better).
Top configs = highest result values (sort descending).

Architecture:
  - ``_SHBracket`` is the internal state machine for a single SH run
    with given (n_configs, start_budget).  Hyperband reuses this.
  - ``SuccessiveHalving`` is the public API class that computes n and r
    from min/max_budget and wraps a single ``_SHBracket``.
"""

from __future__ import annotations

import math
from collections import deque

import numpy as np
from ConfigSpace import Configuration, ConfigurationSpace
from hpo_algorithm import HPOAlgorithm
from utils import sample_config, config_key


# =========================================
# Internal bracket (reused by Hyperband)
# =========================================

class _SHBracket:
    """
    A single Successive Halving bracket.

    Manages one run from ``n_configs`` initial configurations at
    ``start_budget``, halving by ``eta`` each rung up to ``max_budget``.

    Parameters
    ----------
    cs : ConfigurationSpace
        Search space.
    n_configs : int
        Number of configurations to start with.
    start_budget : float
        Budget for the first rung.
    max_budget : float
        Maximum budget (last rung).
    eta : int
        Halving ratio (e.g. 3 → keep top 1/3 each rung).
    rng : np.random.RandomState
        Random state for sampling configs.
    """

    def __init__(
        self,
        cs: ConfigurationSpace,
        n_configs: int,
        start_budget: float,
        max_budget: float,
        eta: int,
        rng: np.random.RandomState,
    ) -> None:
        self._cs = cs
        self._eta = eta
        self._rng = rng

        # Compute rung budgets: start_budget, start_budget*eta, ..., <= max_budget
        #
        # KNOWN LIMITATION: if max_budget is not an exact power-of-eta multiple
        # of start_budget, the last rung will be BELOW max_budget.
        # Example — NB301 (start=1, max=98, η=3):
        #   rungs = [1, 3, 9, 27, 81]  ← tops out at 81, not 98
        # This means SH never evaluates at full fidelity on NB301, which
        # partly explains its lower final accuracy vs Random Search/Hyperband.
        # Hyperband is unaffected: its bracket start budgets (r_s = max·η^{-s})
        # are chosen so the geometric sequence exactly hits max_budget.
        self._rung_budgets: list[float] = []
        b = start_budget
        while b <= max_budget + 1e-9:  # small epsilon for float comparison
            self._rung_budgets.append(float(min(b, max_budget)))
            b *= eta
        if not self._rung_budgets:
            self._rung_budgets = [max_budget]

        # Sample initial configs
        configs = [sample_config(cs, rng) for _ in range(n_configs)]

        # Pending: (config, budget) pairs ready to be returned via ask()
        self._pending: deque[tuple[Configuration, float]] = deque()
        for c in configs:
            self._pending.append((c, self._rung_budgets[0]))

        # Current rung tracking
        self._rung_idx = 0
        self._rung_configs: list[Configuration] = list(configs)  # configs in current rung
        self._results: dict = {}  # config_key → result for current rung
        self._exhausted = False

    @property
    def exhausted(self) -> bool:
        """True when all rungs are done and no pending work remains."""
        return self._exhausted and len(self._pending) == 0

    def ask(self) -> tuple[Configuration, float]:
        """
        Return the next (config, budget) pair to evaluate.

        Raises StopIteration when the bracket is fully exhausted.
        """
        if not self._pending:
            if self._exhausted:
                raise StopIteration("Bracket exhausted")
            # Should not happen if tell() drives promotion correctly
            raise StopIteration("Bracket waiting for tell() results")
        return self._pending.popleft()

    def tell(self, config: Configuration, result: float, budget: float) -> None:
        """
        Record the result for ``config`` at ``budget``.

        When all configs in the current rung have reported, promotes the
        top 1/η fraction to the next rung.
        """
        key = config_key(config)
        self._results[key] = result

        # Check if current rung is complete
        if len(self._results) < len(self._rung_configs):
            return  # still waiting for more results

        # ----- Rung complete: promote top configs -----
        n_current = len(self._rung_configs)
        n_promote = max(1, math.floor(n_current / self._eta))

        # Sort by result DESCENDING (maximize accuracy)
        ranked = sorted(
            self._rung_configs,
            key=lambda c: self._results[config_key(c)],
            reverse=True,
        )
        promoted = ranked[:n_promote]

        # Advance to next rung
        self._rung_idx += 1

        if self._rung_idx < len(self._rung_budgets):
            next_budget = self._rung_budgets[self._rung_idx]
            self._rung_configs = promoted
            self._results = {}
            for c in promoted:
                self._pending.append((c, next_budget))
        else:
            # All rungs done
            self._exhausted = True


# =========================================
# Public API class
# =========================================

class SuccessiveHalving(HPOAlgorithm):
    """
    Successive Halving optimiser.

    Computes the number of initial configurations ``n`` from min/max_budget
    using the standard budget-aware formula from the Hyperband paper, then
    delegates to an internal ``_SHBracket``.

    When the bracket is exhausted, it restarts with a fresh batch of random
    configs (the assignment loop keeps calling ask until total_budget is
    consumed externally).

    Parameters
    ----------
    eta : int
        Halving ratio (default 3).
    seed : int
        Random seed for reproducibility.
    """

    def __init__(
        self,
        cs: ConfigurationSpace,
        total_budget: int,
        min_budget: float,
        max_budget: float,
        eta: int = 3,
        seed: int = 0,
    ) -> None:
        super().__init__(cs, total_budget, min_budget, max_budget)
        self._eta = eta
        self._rng = np.random.RandomState(seed)

        # Number of rungs
        self._s_max = math.floor(
            math.log(max_budget / min_budget) / math.log(eta)
        )

        # Number of initial configs (budget-aware formula from HB paper)
        self._n_initial = math.ceil(
            (eta ** (self._s_max + 1)) / (self._s_max + 1)
        )

        # Create first bracket
        self._bracket = self._new_bracket()

    def _new_bracket(self) -> _SHBracket:
        """Create a fresh SH bracket with a new batch of random configs."""
        return _SHBracket(
            cs=self.cs,
            n_configs=self._n_initial,
            start_budget=self.min_budget,
            max_budget=self.max_budget,
            eta=self._eta,
            rng=self._rng,
        )

    def ask(self) -> tuple[Configuration, float]:
        """Return the next config to evaluate at the appropriate budget."""
        try:
            return self._bracket.ask()
        except StopIteration:
            # Bracket exhausted → restart with fresh configs
            self._bracket = self._new_bracket()
            return self._bracket.ask()

    def tell(self, config: Configuration, result: float, budget: float) -> None:
        """Forward result to the internal bracket for rung promotion."""
        self._bracket.tell(config, result, budget)
