"""
cmaes_runner.py
===============
Runs a single CMA-ES optimisation episode using modcma, detecting restart
conditions and delegating population-size decisions to a pluggable
PopulationController.

Responsibilities
----------------
This module owns everything that is *common across all restart strategies*:

1.  Constructing and configuring the modcma ModularCMAES / Parameters objects.
2.  Stepping the CMA-ES until modcma's internal termination criteria fire.
3.  Detecting *that* a restart is needed (via modcma's standard criteria:
    max_iter, equalfunvalues, flat_fitness, tolx, tolupsigma, conditioncov,
    noeffectaxis, noeffectcoor, stagnation).
4.  Building a RestartState snapshot and calling the controller's
    `select_lambda` to obtain the new population size.
5.  Re-initialising the CMA-ES with the chosen lambda and a fresh sigma,
    while preserving cumulative budget accounting.
6.  Enforcing the global budget limit and returning a RunResult.

What this module does NOT own
------------------------------
-   How to respond to a restart (that is the controller's job).
-   Any controller-specific logic (IPOP doubling, PSA estimation, DRL
    policy forward passes, etc.).

This separation means a DRL controller can be plugged in by simply
passing a different PopulationController subclass to `CMAESRunner`.

Restart stop conditions
-----------------------
The conditions below are implemented by modcma and are *identical* across
all restart strategies considered (IPOP, PSA-CMA-ES, DRL, …).  They are
therefore kept here in the runner, not in any individual controller.

    max_iter        Too many iterations since last restart.
    equalfunvalues  Best fitness unchanged for nbin generations.
    flat_fitness    Many function evaluations returned the same value.
    tolx            Step size / evolution path effectively zero.
    tolupsigma      Step size has grown too large (divergence).
    conditioncov    Condition number of covariance matrix too large.
    noeffectaxis    No movement along principal axes.
    noeffectcoor    No movement in any coordinate direction.
    stagnation      No improvement over a long window.

References
----------
Hansen, N. (2016). The CMA evolution strategy: A tutorial.
    arXiv:1604.00772.
Auger, A. & Hansen, N. (2005). A restart CMA evolution strategy with
    increasing population size. IEEE CEC 2005.
Hansen, N. et al. (2021). COCO: A platform for comparing continuous
    optimizers in a black-box setting. Optimization Methods and Software.
"""

from __future__ import annotations

import math
import time
import warnings
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

warnings.filterwarnings("ignore")

from modcma import ModularCMAES, Parameters

from population_controller import PopulationController, RestartState


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class RunResult:
    """
    Complete record of one CMA-ES run (possibly with multiple restarts).

    Attributes
    ----------
    func_name : str
        Name of the optimisation target (e.g. 'Sphere').
    fid : int
        BBOB function ID.
    iid : int
        BBOB instance ID.
    run_index : int
        Which independent repeat this is (0-indexed).
    controller_name : str
        Name of the PopulationController used (e.g. 'IPOPController').
    dim : int
        Problem dimensionality.
    budget : int
        Total function evaluation budget allocated.
    fopt : float
        Best f(x) - f* achieved.
    used_budget : int
        Function evaluations actually consumed.
    n_restarts : int
        Number of restarts that occurred.
    lambda_history : List[int]
        Population size used in each phase (length = n_restarts + 1).
    restart_criteria_history : List[Dict[str, bool]]
        Which termination criteria fired at each restart event.
    runtime_s : float
        Wall-clock time in seconds.
    converged : bool
        True if fopt < convergence_target (default 1e-8, COCO convention).
    """
    func_name: str
    fid: int
    iid: int
    run_index: int
    controller_name: str
    controller_label: str   # human-readable label from the registry
    dim: int
    budget: int
    fopt: float
    used_budget: int
    n_restarts: int
    lambda_history: List[int] = field(default_factory=list)
    restart_criteria_history: List[Dict[str, bool]] = field(default_factory=list)
    runtime_s: float = 0.0
    converged: bool = False


# ---------------------------------------------------------------------------
# Module-level config array builder
# ---------------------------------------------------------------------------

# modcma module order (from Parameters.__modules__):
#   active, elitist, orthogonal, sequential, threshold_convergence,
#   step_size_adaptation, mirrored, base_sampler, weights_option,
#   local_restart, bound_correction

# local_restart options: (None, 'restart', 'IPOP', 'BIPOP', 'STOP')
# We force local_restart = 'STOP' (index 4) so that modcma halts when its
# internal criteria fire rather than performing its own restart logic.
# The runner then takes over: it calls the controller and re-initialises.
_LOCAL_RESTART_STOP_IDX = 4   # 'STOP' – halts on criteria, no self-restart
_LOCAL_RESTART_NONE_IDX = 0   # 'None' – criteria are never computed

# Position of local_restart in __modules__
_LR_POS = list(Parameters.__modules__).index("local_restart")


def _make_config_array(use_stop: bool = True) -> np.ndarray:
    """
    Build a modcma config array with all defaults except local_restart.

    Parameters
    ----------
    use_stop : bool
        If True, set local_restart='STOP' so modcma computes termination
        criteria and halts when they fire (runner then handles restart).
        If False, set local_restart=None (no criteria computed; useful for
        a pure budget-limited run with no restarts).
    """
    n = len(Parameters.__modules__)
    cfg = np.zeros(n, dtype=int)
    cfg[_LR_POS] = _LOCAL_RESTART_STOP_IDX if use_stop else _LOCAL_RESTART_NONE_IDX
    return cfg


# ---------------------------------------------------------------------------
# CMAESRunner
# ---------------------------------------------------------------------------

class CMAESRunner:
    """
    Executes one complete CMA-ES optimisation episode.

    The runner steps through a series of CMA-ES *phases* (restarts).
    Within each phase it steps modcma until either:
      (a) modcma's standard termination criteria fire  →  ask controller
          for a new lambda, then start a new phase, OR
      (b) the global budget is exhausted               →  terminate.

    Parameters
    ----------
    controller : PopulationController
        Any controller implementing `select_lambda(RestartState) -> int`.
        Pass an IPOPController for the IPOP baseline, or a DRL policy
        subclass for the learned controller.
    dim : int
        Problem dimensionality.
    budget : int
        Maximum total function evaluations.
    sigma0 : float
        Initial step size for every phase.
    convergence_target : float
        Run is considered successful if fopt drops below this value.
        The COCO/BBOB convention is 1e-8 (Hansen et al., 2021).
    enable_restarts : bool
        If False the runner never triggers a restart regardless of criteria.
        Useful for the plain CMA-ES baseline.
    max_restarts : int or None
        Hard cap on number of restarts.  None = unlimited.
    """

    def __init__(
        self,
        controller: PopulationController,
        dim: int = 10,
        budget: int = 100_000,
        sigma0: float = 2.0,
        convergence_target: float = 1e-8,
        enable_restarts: bool = True,
        max_restarts: Optional[int] = None,
    ) -> None:
        self.controller = controller
        self.dim = dim
        self.budget = budget
        self.sigma0 = sigma0
        self.convergence_target = convergence_target
        self.enable_restarts = enable_restarts
        self.max_restarts = max_restarts

    # ------------------------------------------------------------------
    def run(
        self,
        fitness_func: Any,
        func_name: str = "unknown",
        fid: int = 0,
        iid: int = 0,
        run_index: int = 0,
        label: str = "",
    ) -> RunResult:
        """
        Optimise `fitness_func` and return a full RunResult.

        Parameters
        ----------
        fitness_func : callable or IOHFunction
            The objective to minimise (must accept a (dim,) array and
            return a scalar float, or be a modcma-compatible IOHFunction).
        func_name : str
            Human-readable name (for logging / result labelling).
        fid : int
            BBOB function ID.
        iid : int
            BBOB instance ID.
        run_index : int
            Index of this independent repeat.

        Returns
        -------
        RunResult
        """
        self.controller.reset()

        # Cumulative tracking across restarts
        total_used: int = 0
        n_restarts: int = 0
        best_fopt: float = float("inf")
        lambda_history: List[int] = []
        criteria_history: List[Dict[str, bool]] = []

        t0 = time.perf_counter()

        # Decide whether to compute termination criteria
        use_stop = self.enable_restarts
        cfg = _make_config_array(use_stop=use_stop)

        # Initial CMA-ES phase
        es, initial_lambda = self._init_es(fitness_func, cfg, lambda_new=None)
        lambda_history.append(initial_lambda)

        while True:
            # ---- step one generation ------------------------------------
            should_continue = es.step()
            phase_fopt = float(es.parameters.fopt)
            if phase_fopt < best_fopt:
                best_fopt = phase_fopt

            # Notify controller every generation (DRL online-update hook)
            step_state = self._build_state(
                es, n_restarts, total_used, best_fopt
            )
            self.controller.observe_step(step_state)

            # ---- global budget check ------------------------------------
            phase_used = int(es.parameters.used_budget)
            cumulative = total_used + phase_used
            if cumulative >= self.budget:
                total_used = cumulative
                break

            # ---- convergence check --------------------------------------
            if best_fopt <= self.convergence_target:
                total_used = cumulative
                break

            # ---- termination criteria / restart -------------------------
            if not should_continue and self.enable_restarts:
                # Record which criteria fired
                fired = {
                    k: bool(v)
                    for k, v in es.parameters.termination_criteria.items()
                    if v
                }
                criteria_history.append(fired)
                total_used += phase_used

                # Hard cap on restarts
                if self.max_restarts is not None and n_restarts >= self.max_restarts:
                    break

                # Ask controller for new population size
                restart_state = self._build_state(
                    es, n_restarts, total_used, best_fopt,
                    triggered_criteria=fired,
                )
                new_lambda = self.controller.select_lambda(restart_state)
                n_restarts += 1

                # Re-initialise for next phase
                es, _ = self._init_es(
                    fitness_func, cfg,
                    lambda_new=new_lambda,
                    used_so_far=total_used,
                )
                lambda_history.append(new_lambda)

            elif not should_continue:
                # No restarts enabled; just stop
                total_used += phase_used
                break

        elapsed = time.perf_counter() - t0

        return RunResult(
            func_name=func_name,
            fid=fid,
            iid=iid,
            run_index=run_index,
            controller_name=repr(self.controller),
            controller_label=label or repr(self.controller),
            dim=self.dim,
            budget=self.budget,
            fopt=best_fopt,
            used_budget=min(total_used, self.budget),
            n_restarts=n_restarts,
            lambda_history=lambda_history,
            restart_criteria_history=criteria_history,
            runtime_s=elapsed,
            converged=(best_fopt <= self.convergence_target),
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _init_es(
        self,
        fitness_func: Any,
        cfg: np.ndarray,
        lambda_new: Optional[int],
        used_so_far: int = 0,
    ) -> Tuple[ModularCMAES, int]:
        """
        Construct a fresh ModularCMAES instance for one phase.

        Parameters
        ----------
        fitness_func : callable
        cfg : np.ndarray
            modcma config array (local_restart already set).
        lambda_new : int or None
            Desired population size.  If None, use modcma default.
        used_so_far : int
            Budget already consumed before this phase.

        Returns
        -------
        es : ModularCMAES
        actual_lambda : int
            Population size actually used.
        """
        params = Parameters.from_config_array(self.dim, cfg)
        params.sigma0 = self.sigma0
        params.budget = self.budget - used_so_far  # remaining budget for this phase

        if lambda_new is not None:
            # Override lambda; mu follows by convention mu = lambda // 2
            params.lambda_ = int(lambda_new)
            params.mu = int(lambda_new) // 2

        # Re-run selection / adaptation init so weights are consistent
        # with the (possibly new) lambda value.
        params.init_selection_parameters()
        params.init_adaptation_parameters()
        params.init_dynamic_parameters()
        params.init_local_restart_parameters()

        es = ModularCMAES(fitness_func, parameters=params)
        return es, int(params.lambda_)

    def _build_state(
        self,
        es: ModularCMAES,
        n_restarts: int,
        used_so_far: int,
        best_fopt: float,
        triggered_criteria: Optional[Dict[str, bool]] = None,
    ) -> RestartState:
        """Build a RestartState from current modcma state."""
        phase_used = int(es.parameters.used_budget)
        cumulative = used_so_far + phase_used
        return RestartState(
            current_lambda=int(es.parameters.lambda_),
            current_sigma=float(es.parameters.sigma),
            used_budget=cumulative,
            remaining_budget=max(0, self.budget - cumulative),
            n_restarts=n_restarts,
            fopt=best_fopt,
            dim=self.dim,
            triggered_criteria=triggered_criteria or {},
        )