from ioh import ProblemType
from modcma import ModularCMAES, Parameters
from typing import Any, Dict, Optional

from controllers import PopulationController, BBOBState


class CMAES:
    """
    Executes complete CMAES optimisations. Binds to a specific population configuration mechanism on init. Same object can be used across multiple function optimisations.

    Parameters
    -
    controller : PopulationController
        Any controller implementing select_lambda(CurrentState) -> int.
    config : Dict = {}
        configuration specifications for the ModularCMAES object
    """

    def __init__(self, controller: PopulationController | None = None, config: Dict = {}):
        self.controller = controller
        self.config = config
        self.config.update({"compute_termination_criteria": True})
        self.success = False


    def run(self, fitness_func: ProblemType):
        """
        Optimise fitness_func for one full cmaes run.

        Parameters
        -
        fitness_func : Callable function
            The objective to minimise (must return a scalar float, or be a modcma-compatible IOHFunction).

        Returns
        -
        steps:
            Number of steps CMAES was run for
        """
        
        fitness_func.reset()
        self.controller.reset()
        steps: int = 0

        # Initial CMA-ES phase
        cmaes = self.make_runner(fitness_func, self.config, lambda_new=None)

        while True:
            # one generation step
            s1 = not cmaes.step()
            s2 = any(cmaes.parameters.termination_criteria.values())
            stop = s1 or s2
            steps += 1
            sample_fopt = float(cmaes.parameters.fopt)

            # Notify controller every generation
            step_state = self.get_BBOBState(cmaes, fitness_func, sample_fopt, steps)
            self.controller.observe_step(step_state)    # redundant

            # modify population size for next run
            lambda_new = self.controller.select_lambda(step_state)
            cmaes.parameters.update_popsize(lambda_new)

            if stop:
                break

        return steps

    def run_restart(self, fitness_func: ProblemType):
        """
        Optimise fitness_func and return a full run Result.

        Parameters
        -
        fitness_func : Callable function
            The objective to minimise (must return a scalar float, or be a modcma-compatible IOHFunction).
        Returns
        -
        steps : int
            total number of steps made
        starts : int
            total number of times cmaes was started/ restarted
        """
        
        fitness_func.reset()
        self.controller.reset()
        steps: int = 0
        starts: int = 1

        # Initial CMA-ES runner
        cmaes = self.make_runner(fitness_func, self.config, lambda_new=None)

        while True:
            stop = not cmaes.step()     # exhausted budget or terminated from modcma side
            restart = any(cmaes.parameters.termination_criteria.values())   # hit restart criteria
            steps += 1      # increment steps used
            sample_fopt = float(cmaes.parameters.fopt)      # note best candidate of current population for state

            # Notify controller every generation
            step_state = self.get_BBOBState(cmaes, fitness_func, sample_fopt, steps)
            self.controller.observe_step(step_state)

            if restart:
                starts += 1
                lambda_new = self.controller.select_lambda(step_state)
                cmaes = self.make_runner(fitness_func, self.config, lambda_new)

            if stop:
                break

        return steps, starts


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
        Construct a fresh ModularCMAES instance for one restart. Automatically deducts function evaluations already exhausted from available budget

        Parameters
        -
        fitness_func : callable
        config : Dict[str, Any]
            modcma configuration of parameters.
        lambda_new : int or None
            Desired population size.  If None, use modcma default.

        Returns
        -
        cmaes : ModularCMAES
        """
        params = Parameters(d = fitness_func.meta_data.n_variables)
        config.update({'budget': params.budget - fitness_func.state.evaluations})
        params.update(config)

        if lambda_new is not None:
            params.update_popsize(lambda_new)

        cmaes = ModularCMAES(fitness_func, parameters=params)
        return cmaes

    
    def get_BBOBState(self, cmaes: ModularCMAES, func: ProblemType, sample_fopt: float, n: int, extra: Dict = {}):
        """
        Construct a BBOBState object from the current state of a CMA-ES run.
        
        Parameters
        -
        cmaes : ModularCMAES
            The active CMA-ES optimizer whose parameters are reported.
        func : ProblemType
            The current IOH BBOB problem.
        sample_fopt : float
            Best fitness among current population of CMAES
        n : int
            The current restart number or step.
        extra : Dict, optional
            Additional user-defined information to include in the returned BBOBState. Defaults to an empty dictionary.

        Returns
        -
        BBOBState
        """
        return BBOBState(cmaes.parameters.lambda_, cmaes.parameters.sigma, func.meta_data.n_variables, sample_fopt, func.state.current_best, n, func.state.evaluations, cmaes.parameters.budget, cmaes.parameters.termination_criteria, extra, func.meta_data.problem_id, func.meta_data.instance)

