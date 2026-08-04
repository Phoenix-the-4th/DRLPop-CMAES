from abc import ABC, abstractmethod
from dataclasses import dataclass
import numpy as np
import torch
from typing import Any, Dict

from agents import Agent


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
    dim : int
        dimensionality of the function to be optimised
    fopt_sample : float
        Best fitness among the current generation
    fopt_best : float
        Best fitness found so far
    n : int
        How many steps/restarts have already occured (0 before the first) (steps for continuous optimisations, restarts for restart based).
    used_budget : int
        Total function evaluations consumed so far.
    total_budget : int
        Total budget asigned.
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
    total_budget: int
    triggered_criteria: Dict[str, bool]
    extra: Dict[str, Any]


@dataclass
class BBOBState(CurrentState):
    fid: int
    iid: int


# Abstract class for controllers
class PopulationController(ABC):
    """
    Abstract interface for all population-size controllers. Controller is stateful across restarts within one optimisation run (e.g. IPOP needs to track the restart count) but is reset between independent optimisations using reset().

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
        Decide the population size using the given state.

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
        pass  # no-op for restart controllers


# IPOP controller
class IPOP(PopulationController):
    """
    IPOP population controller (Auger, A. and Hansen, N., 2005, September. A restart CMA evolution strategy with increasing population size. In 2005 IEEE congress on evolutionary computation (Vol. 2, pp. 1769-1776). IEEE.).

    After each restart the population size is multiplied by ipop_factor(default 2).

    Parameters
    -
    ipop_factor : int
        Multiplicative increase applied to lambda at each restart. The default value is 2.
    """

    def __init__(
        self,
        ipop_factor: int = 2,
    ):
        super().__init__(restart = True)
        if ipop_factor < 2:
            raise ValueError("ipop_factor must be >= 2")
        self.ipop_factor = ipop_factor

    def reset(self) -> None:
        """Reset for a new independent run."""
        pass

    def select_lambda(self, state: CurrentState) -> int:
        """
        Double the population size.

        Parameters
        -
        state : CurrentState

        Returns
        -
        int
            New lambda for the next CMA-ES run.
        """
        return state.current_lambda * self.ipop_factor

    def __repr__(self) -> str:
        return (
            f"IPOP(ipop_factor={self.ipop_factor})"
        )


# Default controller
class DefaultPop(PopulationController):
    """
    Default population, no restart. Baseline run.

    Parameters
    -
    """

    def __init__(self):
        super().__init__(restart = False)

    def reset(self) -> None:
        """Reset for a new independent run."""
        pass

    def select_lambda(self, state: CurrentState) -> int:
        """
        Poulation size.

        Parameters
        -
        state : CurrentState

        Returns
        -
        int
            New lambda.
        """
        return state.current_lambda

    def __repr__(self) -> str:
        return (
            f"DefaultPop()"
        )


# DRL controller
class DRLPop(PopulationController):
    """
    Deep reinforcement learning based population controller.

    Parameters
    -
    model_file : str
        Path to the model weights to be loaded.
    """

    def __init__(
        self,
        model_file: str,
        state_type: int,
        lb: int,
        ub: int,
    ):
        super().__init__(restart = False)
        self.model_file = model_file
        if state_type == 1:
            obs_dim = 3
        self.model: Agent = Agent(obs_dim, 1)
        self.load_weights()
        self.state_type = state_type
        self.lb = lb
        self.ub = ub

    def reset(self) -> None:
        """Reset for a new independent run."""
        pass

    def select_lambda(self, state: CurrentState) -> int:
        """
        Evaluate next action using the model.

        Parameters
        -
        state : CurrentState

        Returns
        -
        int
            New lambda for the next CMA-ES run.
        """
        with torch.no_grad():
            if self.state_type == 1:
                obs = [state.current_lambda, state.current_sigma, state.used_budget/state.total_budget]
            action = self.model.get_action_mean(torch.as_tensor(obs, dtype=torch.float32))
        newact = 2 + (action - self.lb) / (self.ub - self.lb) * 512
        clipact = np.clip(newact, 2, 512)
        return int(clipact)

    def __repr__(self) -> str:
        return (
            f"DRLController(model={self.model_file})"
        )

    def load_weights(self):
        self.model = self.model.to("cuda" if torch.cuda.is_available() else "cpu")
        self.model.load_state_dict(torch.load(self.model_file, weights_only=True))
        self.model.eval()