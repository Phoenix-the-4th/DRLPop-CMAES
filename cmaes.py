"""
Runs a single CMA-ES optimisation episode using modcma, detecting restart
conditions and delegating population-size decisions to a pluggable
PopulationController.
"""

from controllers import PopulationController, CurrentState, BBOBState, DefaultPop
from dataclasses import dataclass, field
from ioh import problem, ProblemType, get_problem
from modcma import ModularCMAES, Parameters
from typing import Any, Dict, List, Optional, Tuple





class CMAES:
    """
    Executes one complete CMA-ES optimisation run, possibly including multiple restarts.

    Parameters
    -
    controller : PopulationController
        Any controller implementing select_lambda(CurrentState) -> int.
    """

    def __init__(self, controller: PopulationController | None = None, config: Dict = {}):
        self.controller = controller
        self.config = config
        self.config.update({"compute_termination_criteria": True})
        self.success = False


    def run(self, fitness_func: ProblemType):
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
        
        fitness_func.reset()
        self.controller.reset()
        steps: int = 0

        # Initial CMA-ES phase
        cmaes = self.make_runner(fitness_func, self.config, lambda_new=None)

        while True:
            # one generation step
            # should_continue = cmaes.step() and not any(cmaes.parameters.termination_criteria.values())
            stop = not cmaes.step() or any(cmaes.parameters.termination_criteria.values())
            steps += 1
            sample_fopt = float(cmaes.parameters.fopt)

            # Notify controller every generation (DRL online-update hook)
            # step_state = self._build_state(cmaes, n_restarts, budget, best_fopt)
            step_state = self.get_BBOBState(cmaes, fitness_func, sample_fopt, steps)
            self.controller.observe_step(step_state)

            # if fitness_func.state.optimum_found:
            #     print(steps)

            # if not should_continue:
            #     print(step_state)
            #     print(fitness_func.state.optimum_found)
            #     print(cmaes.parameters.termination_criteria)

            if stop:
                break

    def run_restart(self, fitness_func: ProblemType):
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
        
        fitness_func.reset()
        self.controller.reset()
        steps: int = 0

        # Initial CMA-ES phase
        cmaes = self.make_runner(fitness_func, self.config, lambda_new=None)

        while True:
            # one generation step
            # should_continue = cmaes.step() and not any(cmaes.parameters.termination_criteria.values())
            stop = not cmaes.step()
            restart = any(cmaes.parameters.termination_criteria.values()) #and not cmaes.parameters.termination_criteria['max_iter']
            steps += 1
            sample_fopt = float(cmaes.parameters.fopt)

            # Notify controller every generation (DRL online-update hook)
            # step_state = self._build_state(cmaes, n_restarts, budget, best_fopt)
            step_state = self.get_BBOBState(cmaes, fitness_func, sample_fopt, steps)
            self.controller.observe_step(step_state)

            if restart:
                lambda_new = self.controller.select_lambda(step_state)
                cmaes = self.make_runner(fitness_func, self.config, lambda_new)

            if stop:
                break


    def __call__(self, func: ProblemType):
        if self.controller.restart:
            self.run_restart(func)
        else:
            self.run(func)
        self.success: bool = func.state.optimum_found

    def make_runner(
        self,
        fitness_func: ProblemType,
        config: Dict[str, Any] = {"compute_termination_criteria": True},
        lambda_new: Optional[int] = None
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
        config.update({'budget': params.budget - fitness_func.state.evaluations})
        params.update(config)

        if lambda_new is not None:
            params.update_popsize(lambda_new)

        cmaes = ModularCMAES(fitness_func, parameters=params)
        return cmaes

    
    def get_BBOBState(self, cmaes: ModularCMAES, func: ProblemType, sample_fopt: float, n: int, extra: Dict = {}):
        return BBOBState(cmaes.parameters.lambda_, cmaes.parameters.sigma, func.meta_data.n_variables, sample_fopt, func.state.current_best, n, func.state.evaluations, cmaes.parameters.budget, cmaes.parameters.termination_criteria, extra, func.meta_data.problem_id, func.meta_data.instance)

