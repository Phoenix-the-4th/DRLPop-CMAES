"""
Runs a single CMA-ES optimisation episode using modcma, detecting restart
conditions and delegating population-size decisions to a pluggable
PopulationController.
"""

from controllers import PopulationController, CurrentState
from dataclasses import dataclass, field
from ioh import problem, ProblemType
from modcma import ModularCMAES, Parameters
from typing import Any, Dict, List, Optional, Tuple

# Framework to store individual run results
@dataclass
class BaseResult:
    """
    Complete record of one CMA-Es run.

    Attributes
    -
    func_name : str
        Name of the optimisation target (e.g. 'Sphere').
    fid : int
        BBOB function ID.
    iid : int
        BBOB instance ID.
    dim : int
        Problem dimensionality.
    func_class : str
        Name of the benchmark from which function is being used.
    controller_name : str
        Name of the PopulationController used (e.g. 'IPOPController').
    run_index : int
        Which independent repeat this is (0-indexed).
    fopt : float
        Best f(x) - f* achieved.
    used_budget : int
        Function evaluations actually consumed.
    n_restarts : int
        Number of restarts that occurred.
    lambda_history : List[int]
        Population size used in each phase (length = n_restarts + 1).
    fopt_history : List[float]
        Optimal phase at the end of each phase.
    restart_criteria_history : List[Dict[str, bool]]
        Which termination criteria fired at each restart event.
    """
    func_class: str
    func_name: str
    fid: int
    iid: int
    dim: int
    controller_name: str
    run_index: int
    fopt_best: float
    total_used_budget: int
    fopt_history_budget: Dict[int, float]
    lambda_history: List[int]

@dataclass
class RestartResult(BaseResult):
    """
    Complete record of one CMA-Es run.

    Attributes
    -
    func_name : str
        Name of the optimisation target (e.g. 'Sphere').
    fid : int
        BBOB function ID.
    iid : int
        BBOB instance ID.
    dim : int
        Problem dimensionality.
    func_class : str
        Name of the benchmark from which function is being used.
    controller_name : str
        Name of the PopulationController used (e.g. 'IPOPController').
    run_index : int
        Which independent repeat this is (0-indexed).
    fopt : float
        Best f(x) - f* achieved.
    used_budget : int
        Function evaluations actually consumed.
    n_restarts : int
        Number of restarts that occurred.
    lambda_history : List[int]
        Population size used in each phase (length = n_restarts + 1).
    fopt_history : List[float]
        Optimal phase at the end of each phase.
    restart_criteria_history : List[Dict[str, bool]]
        Which termination criteria fired at each restart event.
    """
    n_restarts: int
    fopt_sample_history: List[float] = field(default_factory=list)
    fopt_best_history: List[float] = field(default_factory=list)
    restart_criteria_history: List[Dict[str, bool]] = field(default_factory=list)

@dataclass
class ContResult(BaseResult):
    """
    Complete record of one CMA-Es run.

    Attributes
    -
    func_name : str
        Name of the optimisation target (e.g. 'Sphere').
    fid : int
        BBOB function ID.
    iid : int
        BBOB instance ID.
    dim : int
        Problem dimensionality.
    func_class : str
        Name of the benchmark from which function is being used.
    controller_name : str
        Name of the PopulationController used (e.g. 'IPOPController').
    run_index : int
        Which independent repeat this is (0-indexed).
    fopt : float
        Best f(x) - f* achieved.
    used_budget : int
        Function evaluations actually consumed.
    n_restarts : int
        Number of restarts that occurred.
    lambda_history : List[int]
        Population size used in each phase (length = n_restarts + 1).
    fopt_history : List[float]
        Optimal phase at the end of each phase.
    restart_criteria_history : List[Dict[str, bool]]
        Which termination criteria fired at each restart event.
    """
    n_steps: int
    fopt_sample_history: List[float] = field(default_factory=list)
    fopt_best_history: List[float] = field(default_factory=list)
    restart_criteria_history: List[Dict[str, bool]] = field(default_factory=list)




class CMAES:
    """
    Executes one complete CMA-ES optimisation run, possibly including multiple restarts.

    Parameters
    -
    controller : PopulationController
        Any controller implementing select_lambda(CurrentState) -> int.
    """

    def __init__(
        self,
        controller: PopulationController
    ):
        self.controller = controller

    def run(
        self,
        fitness_func: ProblemType,
        func_class: str,
        run_index: int = 0,
    ) -> RestartResult:
        """
        Optimise fitness_func and return a full run Result.

        Parameters
        -
        fitness_func : Callable function
            The objective to minimise (must return a scalar float, or be a modcma-compatible IOHFunction).
        func_name : str
            Name of the optimisation target (e.g. 'Sphere').
        fid : int
            BBOB function ID.
        iid : int
            BBOB instance ID.
        dim : int
            Problem dimensionality.
        func_class : str
            Name of the benchmark from which function is being used.
        run_index : int
            Index of this independent repeat.

        Returns
        -
        Result object
        """
        self.controller.reset()
        
        func_name = fitness_func.meta_data.name
        fid = fitness_func.meta_data.problem_id
        iid = fitness_func.meta_data.instance
        dim = fitness_func.meta_data.n_variables

        # Cumulative tracking across restarts
        total_budget_used: int = 0
        n_restarts: int = 0
        best_fopt: float = float("inf")
        lambda_history: List[int] = []
        fopt_history: List[int] = []
        criteria_history: List[Dict[str, bool]] = []

        # FIX: config
        # Initial CMA-ES phase
        cmaes = self.make_runner(fitness_func, getattr(self.controller, 'config', {}), lambda_new=None)
        lambda_history.append(cmaes.parameters.lambda_)

        while True:
            # one generation step
            should_continue = cmaes.step()
            phase_fopt = float(cmaes.parameters.fopt)
            best_fopt = min(best_fopt, phase_fopt)
            fopt_history.append(best_fopt)

            # Notify controller every generation (DRL online-update hook)
            step_state = self._build_state(
                cmaes, n_restarts, total_budget_used, best_fopt
            )
            self.controller.observe_step(step_state)

            # global budget check
            phase_used = int(cmaes.parameters.used_budget)
            cumulative = total_budget_used + phase_used
            if cumulative >= cmaes.parameters.budget:
                total_used = cumulative
                break

            # termination criteria / restart
            if not should_continue and self.controller.restart:
                # Record which criteria fired
                fired = {
                    k: bool(v)
                    for k, v in cmaes.parameters.termination_criteria.items()
                    if v
                }
                criteria_history.append(fired)
                total_used += phase_used

                # Ask controller for new population size
                restart_state = self._build_state(
                    cmaes, n_restarts, total_used, best_fopt,
                    triggered_criteria=fired,
                )
                new_lambda = self.controller.select_lambda(restart_state)
                n_restarts += 1

                # Re-initialise for next phase
                cmaes, _ = self.make_runner(
                    fitness_func, getattr(self.controller, 'config', {}),
                    lambda_new=new_lambda,
                    used_so_far=total_used,
                )
                lambda_history.append(new_lambda)

            elif not should_continue:
                # No restarts enabled; just stop
                total_used += phase_used
                break


        return RestartResult(
            func_name = func_name,
            fid = fid,
            iid = iid,
            dim = dim,
            func_class = func_class,
            controller_name=repr(self.controller),
            run_index=run_index,
            fopt_best=best_fopt,
            total_used_budget=total_used,
            n_restarts=n_restarts,
            lambda_history=lambda_history,
            fopt_best_history=fopt_history,
            restart_criteria_history=criteria_history,
            fopt_history_budget={}
        )

    def run_verb(
        self,
        fitness_func: ProblemType,
        func_class: str,
        run_index: int = 0,
    ) -> RestartResult:
        """
        Optimise fitness_func and return a full run Result.

        Parameters
        -
        fitness_func : Callable function
            The objective to minimise (must return a scalar float, or be a modcma-compatible IOHFunction).
        func_name : str
            Name of the optimisation target (e.g. 'Sphere').
        fid : int
            BBOB function ID.
        iid : int
            BBOB instance ID.
        dim : int
            Problem dimensionality.
        func_class : str
            Name of the benchmark from which function is being used.
        run_index : int
            Index of this independent repeat.

        Returns
        -
        Result object
        """
        self.controller.reset()
        
        func_name = fitness_func.meta_data.name
        fid = fitness_func.meta_data.problem_id
        iid = fitness_func.meta_data.instance
        dim = fitness_func.meta_data.n_variables

        # Cumulative tracking across restarts
        total_budget_used: int = 0
        n_restarts: int = 0
        best_fopt: float = float("inf")
        lambda_history: List[int] = []
        fopt_history: List[int] = []
        criteria_history: List[Dict[str, bool]] = []

        # FIX: config
        # Initial CMA-ES phase
        cmaes = self.make_runner(fitness_func, getattr(self.controller, 'config', {}), lambda_new=None)
        lambda_history.append(cmaes.parameters.lambda_)

        while True:
            # one generation step
            should_continue = cmaes.step()
            phase_fopt = float(cmaes.parameters.fopt)
            best_fopt = min(best_fopt, phase_fopt)
            fopt_history.append(best_fopt)

            # Notify controller every generation (DRL online-update hook)
            step_state = self._build_state(
                cmaes, n_restarts, total_budget_used, best_fopt
            )
            self.controller.observe_step(step_state)

            # global budget check
            phase_used = int(cmaes.parameters.used_budget)
            cumulative = total_budget_used + phase_used
            if cumulative >= cmaes.parameters.budget:
                total_used = cumulative
                break

            # termination criteria / restart
            if not should_continue and self.controller.restart:
                # Record which criteria fired
                fired = {
                    k: bool(v)
                    for k, v in cmaes.parameters.termination_criteria.items()
                    if v
                }
                criteria_history.append(fired)
                total_used += phase_used

                # Ask controller for new population size
                restart_state = self._build_state(
                    cmaes, n_restarts, total_used, best_fopt,
                    triggered_criteria=fired,
                )
                new_lambda = self.controller.select_lambda(restart_state)
                n_restarts += 1

                # Re-initialise for next phase
                cmaes, _ = self.make_runner(
                    fitness_func, getattr(self.controller, 'config', {}),
                    lambda_new=new_lambda,
                    used_so_far=total_used,
                )
                lambda_history.append(new_lambda)

            elif not should_continue:
                # No restarts enabled; just stop
                total_used += phase_used
                break


        return RestartResult(
            func_name = func_name,
            fid = fid,
            iid = iid,
            dim = dim,
            func_class = func_class,
            controller_name=repr(self.controller),
            run_index=run_index,
            fopt_best=best_fopt,
            total_used_budget=total_used,
            n_restarts=n_restarts,
            lambda_history=lambda_history,
            fopt_best_history=fopt_history,
            restart_criteria_history=criteria_history,
            fopt_history_budget={}
        )

    def __call__(self, *args, **kwds):
        self.run_verb(args[0], "BBOB")
        pass

    def make_runner(
        self,
        fitness_func: Any,
        config: Optional[Dict[str, Any]],
        lambda_new: Optional[int],
        used_so_far: int = 0,
    ) -> ModularCMAES:
        """
        Construct a fresh ModularCMAES instance for one restart.

        Parameters
        -
        fitness_func : callable
        config : Dict[str, Any]
            modcma configuration of parameters.
        lambda_new : int or None
            Desired population size.  If None, use modcma default.
        used_so_far : int
            Budget already consumed before this phase.

        Returns
        -
        cmaes : ModularCMAES
        actual_lambda : int
            Population size actually used.
        """
        params = Parameters(d = fitness_func.meta_data.n_variables)
        params.update(config)

        if lambda_new is not None:
            params.update_popsize(lambda_new)

        cmaes = ModularCMAES(fitness_func, parameters=params)
        return cmaes

    def _build_state(
        self,
        cmaes: ModularCMAES,
        n_restarts: int,
        used_so_far: int,
        best_fopt: float,
        triggered_criteria: Optional[Dict[str, bool]] = None,
    ) -> CurrentState:
        """Build a CurrentState from current modcma state."""
        phase_used = int(cmaes.parameters.used_budget)
        cumulative = used_so_far + phase_used
        return CurrentState(
            current_lambda=int(cmaes.parameters.lambda_),
            current_sigma=float(cmaes.parameters.sigma),
            used_budget=cumulative,
            dim = 2,
            n=n_restarts,
            fopt_best=best_fopt,
            fopt_sample=best_fopt,
            triggered_criteria=triggered_criteria or {},
        )






