"""
population_controller.py
========================
Defines the abstract PopulationController interface and concrete
implementations.

Architecture
------------
A PopulationController is responsible for ONE decision only:
    Given that the CMA-ES has met a restart criterion, what should the
    new population size (lambda) be for the next run?

Everything else (detecting *when* to restart, resetting internal CMA-ES
state, tracking budget) is handled by the CMAESRunner in cmaes_runner.py.
This clean boundary makes it trivial to swap in a DRL controller later
without touching any CMA-ES mechanics.

Restart-triggering conditions (equalfunvalues, tolx, stagnation, …) are
*common across all algorithms* and belong in the runner.  What differs
between IPOP, PSA-CMA-ES, a DRL policy, etc. is only the mapping
    (state) -> new_lambda
and that is precisely what each subclass here encapsulates.

Included controllers
--------------------
IPOPController
    Doubles lambda after every restart (Auger & Hansen, 2005).
    Population size grows as lambda_0 * 2^k where k is the restart index,
    capped at max_lambda.

To plug in a DRL controller, subclass PopulationController and implement
`select_lambda`.  The runner will call it automatically at every restart
event.

References
----------
Auger, A. & Hansen, N. (2005). A restart CMA evolution strategy with
    increasing population size. IEEE CEC 2005.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, Optional


# ---------------------------------------------------------------------------
# State snapshot passed to every controller at restart time
# ---------------------------------------------------------------------------

@dataclass
class RestartState:
    """
    Snapshot of CMA-ES state at the moment a restart is triggered.

    This is the observation a controller receives.  A DRL policy will use
    the same dataclass; unused fields can simply be ignored.

    Attributes
    ----------
    current_lambda : int
        Population size of the run that just terminated.
    current_sigma : float
        Step size at termination.
    used_budget : int
        Total function evaluations consumed so far (across all restarts).
    remaining_budget : int
        Budget still available.
    n_restarts : int
        How many restarts have already occurred (0-indexed: 0 before the
        first restart).
    fopt : float
        Best function value (f - f*) found so far.
    dim : int
        Problem dimensionality.
    triggered_criteria : Dict[str, bool]
        Which of the standard stop conditions fired this restart.
        Keys match modcma's termination_criteria dict:
        'max_iter', 'equalfunvalues', 'flat_fitness', 'tolx',
        'tolupsigma', 'conditioncov', 'noeffectaxis', 'noeffectcoor',
        'stagnation'.
    extra : dict
        Arbitrary extra information; reserved for DRL feature engineering.
    """
    current_lambda: int
    current_sigma: float
    used_budget: int
    remaining_budget: int
    n_restarts: int
    fopt: float
    dim: int
    triggered_criteria: Dict[str, bool] = field(default_factory=dict)
    extra: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------

class PopulationController(ABC):
    """
    Abstract interface for all population-size controllers.

    A controller is stateful across restarts within one run (e.g. IPOP
    needs to track the restart count) but is reset between independent runs
    via `reset()`.
    """

    @abstractmethod
    def reset(self) -> None:
        """Reset internal state for a new independent run."""
        ...

    @abstractmethod
    def select_lambda(self, state: RestartState) -> int:
        """
        Decide the population size for the next CMA-ES restart.

        Parameters
        ----------
        state : RestartState
            Snapshot of the CMA-ES state at the moment a restart is triggered.

        Returns
        -------
        int
            The new population size lambda >= 2.
        """
        ...

    def observe_step(self, state: RestartState) -> None:
        """
        Optional hook called every CMA-ES *generation* (not just at restart).

        DRL controllers that update online can override this.  The default
        implementation is a no-op, so IPOP and other analytic controllers
        are unaffected.

        Parameters
        ----------
        state : RestartState
            Current state (used_budget, fopt, etc. are updated each generation).
        """
        pass  # no-op for analytic controllers


# ---------------------------------------------------------------------------
# IPOP controller
# ---------------------------------------------------------------------------

class IPOPController(PopulationController):
    """
    IPOP population controller (Auger & Hansen, 2005).

    After each restart the population size is multiplied by `ipop_factor`
    (default 2), so successive runs use populations of size
        lambda_0, lambda_0*2, lambda_0*4, ...
    capped at `max_lambda`.

    This increasing-population strategy gives the algorithm a progressively
    broader view of the search space after each failed attempt, helping it
    escape local optima on multimodal problems (Auger & Hansen, 2005).

    Parameters
    ----------
    ipop_factor : int
        Multiplicative increase applied to lambda at each restart.
        The canonical value is 2 (Auger & Hansen, 2005).
    max_lambda : int
        Hard upper bound on population size.  modcma uses mu < 512, which
        corresponds to lambda < 1024 with the default mu = lambda // 2.
        We match that convention here.
    initial_lambda : int or None
        Starting population size.  If None, the runner supplies the
        modcma default (4 + floor(3 * ln(d))).

    References
    ----------
    Auger, A. & Hansen, N. (2005). A restart CMA evolution strategy with
        increasing population size. IEEE CEC 2005.
    """

    def __init__(
        self,
        ipop_factor: int = 2,
        max_lambda: int = 512,
        initial_lambda: Optional[int] = None,
    ) -> None:
        if ipop_factor < 2:
            raise ValueError("ipop_factor must be >= 2")
        self.ipop_factor = ipop_factor
        self.max_lambda = max_lambda
        self.initial_lambda = initial_lambda

        # Internal state (reset between runs)
        self._restart_count: int = 0
        self._last_lambda: Optional[int] = None

    # ------------------------------------------------------------------
    def reset(self) -> None:
        """Reset for a new independent run."""
        self._restart_count = 0
        self._last_lambda = None

    # ------------------------------------------------------------------
    def select_lambda(self, state: RestartState) -> int:
        """
        Double the population size, capped at max_lambda.

        On the first restart the current lambda is taken from `state`
        (which reflects the modcma default if initial_lambda was None).

        Parameters
        ----------
        state : RestartState

        Returns
        -------
        int
            New lambda for the next CMA-ES run.
        """
        if self._last_lambda is None:
            # First restart: use whatever the runner started with
            base = state.current_lambda
        else:
            base = self._last_lambda

        new_lambda = min(base * self.ipop_factor, self.max_lambda)
        self._restart_count += 1
        self._last_lambda = new_lambda
        return new_lambda

    # ------------------------------------------------------------------
    def __repr__(self) -> str:
        return (
            f"IPOPController(ipop_factor={self.ipop_factor}, "
            f"max_lambda={self.max_lambda})"
        )