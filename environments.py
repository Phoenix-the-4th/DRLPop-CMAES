import numpy as np
import gymnasium as gym
from enum import Enum
from itertools import product
from cmaes import CMAES
from typing import List

class StateType(Enum):
    PSB = 1     # population size, step size, fraction of budget used

class RewardType(Enum):
    FBEST_IMP = 1       # absolute improvement in best f-value
    FBEST_IMP_RATIO = 2 # relative improvement in best f-value
    FVAL_IMP_RATIO = 3  # relative improvement in current population f-value


class CMAEnv(gym.Env):
    """
    Single-generation CMA-ES gymnasium environment for continuous PPO.

    Episode : one complete BBOB problem run (budget exhausted or optimum found).
    Step    : one CMA-ES generation. The agent sets the population size lambda for the next generation; the environment evaluates that generation and returns the resulting state and reward.

    Observation StateType.PSB: [lambda, sigma, evaluations / budget]

    Action Box: Desired population size as a float in [2, max_lambda]. Clipped and rounded to an integer inside step().

    Reward (configurable via RewardType):
        FBEST_IMP       : last_fbest - fbest
        FBEST_IMP_RATIO : (last_fbest - fbest) / (|last_fbest| + eps)
        FVAL_IMP_RATIO  : (last_fval  - fval)  / (|last_fval|  + eps)

    terminated : True when modcma termination criteria fire OR budget is exhausted.
    truncated  : True when the global optimum is found within tolerance.
    """

    def __init__(self,
                 dims: List[int] = [10],
                 fids: List[int] = list(range(1, 25)),
                 iids: List[int] = list(range(15)),
                 state_type: StateType = StateType.PSB,
                 reward_type: RewardType = RewardType.FBEST_IMP,
                 max_lambda: int = np.iinfo(np.int64).max,
                 fp = np.float64):
        super().__init__()

        self.problems = list(product(fids, iids, dims))
        np.random.shuffle(self.problems)
        self.factory = CMAES()
        self.state_type = state_type
        self.reward_type = reward_type
        self.max_lambda = max_lambda

        # action space: continuous float so Normal-distribution PPO can act
        # step() rounds and clips to [2, max_lambda] before passing to update_popsize
        self.action_space = gym.spaces.Box(low=np.float64(2), high=np.float64(max_lambda), shape=(1,), dtype=np.float64)

        if self.state_type == StateType.PSB:
            self.observation_space = gym.spaces.Box(low=0, high=np.finfo(fp).max, shape=(3,), dtype=fp)

        # episode state — initialised in reset()
        self.pid = None
        self.cmaes: ModularCMAES | None = None
        self.last_fval = np.inf
        self.last_fbest = np.inf
        self.fval = np.inf
        self.fbest = np.inf
        self.terminated = False
        self.truncated = False

    def reset(self, *, seed = None, options = None):
        super().reset(seed=seed, options=options)

        if self.pid is None:
            self.pid = 0
        else:
            self.pid = (self.pid + 1) % len(self.problems)
            if self.pid == 0:
                np.random.shuffle(self.problems)

        func = get_problem(*self.problems[self.pid])
        self.cmaes = self.factory.make_runner(func)

        # Run the first generation so the state is meaningful on the first step
        should_continue = self.cmaes.step()
        self._update(should_continue)

        # gymnasium API: return (obs, info)
        return self._state(), {}

    # ------------------------------------------------------------------
    def step(self, action):
        # Convert continuous float action to a valid integer population size
        lambda_new = int(np.clip(np.round(float(action[0])), 2, self.max_lambda))

        if lambda_new != self.cmaes.parameters.lambda_:
            self.cmaes.parameters.update_popsize(lambda_new)

        should_continue = self.cmaes.step()
        self._update(should_continue)

        # gymnasium API: return (obs, reward, terminated, truncated, info)
        return self._state(), self._reward(), self.terminated, self.truncated, {}

    def _reward(self):
        eps = np.finfo(np.float32).eps
        if self.reward_type == RewardType.FBEST_IMP:
            return self.last_fbest - self.fbest
        elif self.reward_type == RewardType.FBEST_IMP_RATIO:
            return (self.last_fbest - self.fbest) / (abs(self.last_fbest) + eps)
        elif self.reward_type == RewardType.FVAL_IMP_RATIO:
            return (self.last_fval - self.fval) / (abs(self.last_fval) + eps)

    def _state(self):
        if self.state_type == StateType.PSB:
            return np.array([self.cmaes.parameters.lambda_, self.cmaes.parameters.sigma, self.cmaes._fitness_func.state.evaluations/self.cmaes.parameters.budget])

    def _update(self, should_continue: bool = True):
        """
        Refresh cached statistics after a CMA-ES generation.

        Parameters
        ----------
        should_continue : bool
            Return value of cmaes.step().  False means the per-phase budget is exhausted — that is not captured by termination_criteria, so we must include it explicitly in the terminated flag.
        """
        self.last_fval = self.fval
        self.last_fbest = self.fbest

        # Guard against empty or all-NaN populations (degenerate edge cases)
        f_vals = np.array(self.cmaes.parameters.population.f, dtype=float)
        if len(f_vals) > 0 and not np.all(np.isnan(f_vals)):
            self.fval = float(np.nanmin(f_vals))
        else:
            self.fval = self.last_fval

        self.fbest = self.cmaes._fitness_func.state.current_best.y

        # terminated = budget gone  OR  any modcma convergence criterion fired
        self.terminated = not should_continue or any(self.cmaes.parameters.termination_criteria.values())
        # truncated = optimum found within the BBOB tolerance
        self.truncated = self.cmaes._fitness_func.state.optimum_found