"""
Defines the abstract PopulationController interface and concrete implementations.

"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, Optional



# State snapshot passed to controller for decision making
@dataclass
class CurrentState:
    """
    Snapshot of CMA-ES state, this is the observation a controller receives.

    Attributes
    -
    current_lambda : int
        Most recently used population size of the run.
    current_sigma : float
        Most recently used step size of the run.
    used_budget : int
        Total function evaluations consumed so far.
    remaining_budget : int
        Budget still available.
    n : int
        How many steps/restarts have already occured (0 before the first) (steps for continuous optimisations, restarts for restart based).
    fopt : float
        Best function value (f - f*) found so far.
    dim : int
        Problem dimensionality.
    triggered_criteria : Dict[str, bool]
        Which of the standard stop conditions fired this restart. Keys match modcma's termination_criteria dict: 'max_iter','equalfunvalues', 'flat_fitness', 'tolx', 'tolupsigma', 'conditioncov', 'noeffectaxis', 'noeffectcoor', 'stagnation'.
    extra : dict
        Arbitrary extra information; reserved for DRL feature engineering.
    """
    current_lambda: int
    current_sigma: float
    dim: int
    fopt_sample: float
    fopt_best: float
    n: int
    used_budget: int
    # triggered_criteria: Dict[str, bool] = field(default_factory=dict)
    # extra: Dict[str, Any] = field(default_factory=dict)
    triggered_criteria: Dict[str, bool]
    extra: Dict[str, Any]


@dataclass
class BBOBState(CurrentState):
    fid: int
    iid: int


##### need to add new data class to set the restart conditions of cmaes, this will replace the boolean restart


# Abstract class for controllers
class PopulationController(ABC):
    """
    Abstract interface for all population-size controllers. Controller is stateful across restarts within one run (e.g. IPOP needs to track the restart count) but is reset between independent runs using reset().

    Parameters
    -
    restart : bool
        Whether current controller is a restart based controller (True) or continuous adaptation controller (False)
    """

    def __init__(self, restart: bool):
        super().__init__()
        self.restart: bool = restart

    @abstractmethod
    def reset(self) -> None:
        """Reset internal state for a new independent run."""
        ...

    @abstractmethod
    def select_lambda(self, state: CurrentState) -> int:
        """
        Decide the population size for the next CMA-ES restart.

        Parameters
        -
        state : CurrentState
            Snapshot of the CMA-ES state.

        Returns
        -
        int
            The new population size lambda >= 2.
        """
        ...

    def observe_step(self, state: CurrentState) -> None:
        """
        Optional hook called every CMA-ES generation.

        DRL controllers that update online can override this.  The default implementation is a no-op, so IPOP and other analytic controllers are unaffected.

        Parameters
        -
        state : CurrentState
            Current state (used_budget, fopt, etc. are updated each generation).
        """
        pass  # no-op for restart controllersPxt4compscis


# IPOP controller
class IPOP(PopulationController):
    """
    IPOP population controller (Auger, A. and Hansen, N., 2005, September. A restart CMA evolution strategy with increasing population size. In 2005 IEEE congress on evolutionary computation (Vol. 2, pp. 1769-1776). IEEE.).

    After each restart the population size is multiplied by ipop_factor(default 2).

    Parameters
    -
    ipop_factor : int
        Multiplicative increase applied to lambda at each restart. The default value is 2.
    initial_lambda : int or None
        Starting population size.  If None, the runner supplies the modcma default (4 + floor(3 * ln(d))).
    """

    def __init__(
        self,
        ipop_factor: int = 2,
        initial_lambda: Optional[int] = None,
    ):
        super().__init__(restart = True)
        if ipop_factor < 2:
            raise ValueError("ipop_factor must be >= 2")
        self.ipop_factor = ipop_factor
        self.initial_lambda = initial_lambda
        self.current_lambda = initial_lambda

    def reset(self) -> None:
        """Reset for a new independent run."""
        self.current_lambda = self.initial_lambda
        pass

    def select_lambda(self, state: CurrentState) -> int:
        """
        Double the population size. On the first restart the current lambda is taken from state (which reflects the modcma default if initial_lambda was None).

        Parameters
        -
        state : CurrentState

        Returns
        -
        int
            New lambda for the next CMA-ES run.
        """
        if self.current_lambda is None:
            # First restart: use whatever the runner started with
            self.current_lambda = state.current_lambda

        self.current_lambda *= self.ipop_factor
        return self.current_lambda

    def __repr__(self) -> str:
        return (
            f"IPOPController(ipop_factor={self.ipop_factor})"
        )


# Default controller
class DefaultPop(PopulationController):
    """
    IPOP population controller (Auger, A. and Hansen, N., 2005, September. A restart CMA evolution strategy with increasing population size. In 2005 IEEE congress on evolutionary computation (Vol. 2, pp. 1769-1776). IEEE.).

    After each restart the population size is multiplied by ipop_factor(default 2).

    Parameters
    -
    ipop_factor : int
        Multiplicative increase applied to lambda at each restart. The default value is 2.
    initial_lambda : int or None
        Starting population size.  If None, the runner supplies the modcma default (4 + floor(3 * ln(d))).
    """

    def __init__(
        self,
        initial_lambda: Optional[int] = None,
    ):
        super().__init__(restart = False)
        self.initial_lambda = initial_lambda

    def reset(self) -> None:
        """Reset for a new independent run."""
        pass

    def select_lambda(self, state: CurrentState) -> int:
        """
        Double the population size. On the first restart the current lambda is taken from state (which reflects the modcma default if initial_lambda was None).

        Parameters
        -
        state : CurrentState

        Returns
        -
        int
            New lambda for the next CMA-ES run.
        """
        return self.current_lambda

    def __repr__(self) -> str:
        return (
            f"DefaultController(lambda={self.initial_lambda})"
        )
