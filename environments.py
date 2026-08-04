from enum import Enum
import gymnasium as gym
from ioh import get_problem
from itertools import product
from modcma import ModularCMAES
import numpy as np
from typing import List

from cmaes import CMAES


class StateType(Enum):
    PSB = 1     # population size, step size, fraction of budget used

class RewardType(Enum):
    FBEST_IMP = 1       # absolute improvement in best f-value
    FBEST_IMP_RATIO = 2 # relative improvement in best f-value
    RAW_BEST = 3        # negative raw best value from ioh function
    FBEST_IMP_EVALS = 4 # improvement in best f-value per eval used


class CMAEnv(gym.Env):
    """
    CMA-ES gymnasium environment for continuous PPO.

    Episode : one complete BBOB problem run (budget exhausted or optimum found).
    Step    : one CMA-ES generation. The agent sets the population size lambda for the next generation; the environment evaluates that generation and returns the resulting state and reward.

    terminated : True when modcma termination criteria fire OR budget is exhausted.
    truncated  : True when the global optimum is found within tolerance.
    """

    def __init__(self,
                 dims: List[int] = [10],
                 fids: List[int] = list(range(1, 25)),
                 iids: List[int] = list(range(15)),
                 state_type: StateType = StateType.PSB,
                 reward_type: RewardType = RewardType.FBEST_IMP_RATIO,
                 max_lambda: int = np.iinfo(np.int64).max,
                 fp = np.float64):
        super().__init__()

        self.problems = list(product(fids, iids, dims))
        np.random.shuffle(self.problems)
        self.cache = dict()
        for problem in self.problems:
            self.cache[problem] = get_problem(*problem)
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
        self.last_fbest = np.inf
        self.fbest = np.inf
        self.terminated = False
        self.truncated = False
        self.episode_returns = 0
        self.episode_length = 0

    def reset(self, *, seed = None, options = None):
        super().reset(seed=seed, options=options)

        if self.pid is None:
            self.pid = 0
        else:
            self.pid = (self.pid + 1) % len(self.problems)
            if self.pid == 0:
                np.random.shuffle(self.problems)

        func = self.cache[self.problems[self.pid]]
        func.reset()
        self.cmaes = self.factory.make_runner(func)
        self.last_fbest = np.inf
        self.fbest = np.inf
        self.terminated = False
        self.truncated = False
        self.episode_returns = 0
        self.episode_length = 0

        # Run the first generation so the state is meaningful on the first step
        should_continue = self.cmaes.step()
        self._update(should_continue)

        # gymnasium API: return (obs, info)
        return self._state(), {}

    def step(self, action):
        # Convert continuous float action to a valid integer population size
        if not self.terminated and not self.truncated:
            lambda_new = int(np.clip(np.round(float(action[0])), 2, self.max_lambda))

            if lambda_new != self.cmaes.parameters.lambda_:
                self.cmaes.parameters.update_popsize(lambda_new)

            should_continue = self.cmaes.step()
            self._update(should_continue)

            # gymnasium API: return (obs, reward, terminated, truncated, info)
            self.episode_length = self.cmaes._fitness_func.state.evaluations
            self.episode_returns += self._reward()
        else:
            print("This should not be happening", file = open("WARNING.txt", 'a'))
        return self._state(), self._reward(), self.terminated, self.truncated, {}

    def _reward(self):
        eps = np.finfo(np.float32).eps
        bonus = 20 if self.truncated else 0
        if self.reward_type == RewardType.FBEST_IMP:
            return (self.last_fbest - self.fbest) + bonus
        elif self.reward_type == RewardType.FBEST_IMP_RATIO:
            return ((self.last_fbest - self.fbest) / (abs(self.last_fbest) + eps)) + bonus
        elif self.reward_type == RewardType.RAW_BEST:
            return (- self.fbest) + bonus ** 3
        elif self.reward_type == RewardType.FBEST_IMP_EVALS:
            return (self.last_fbest - self.fbest) / ((abs(self.last_fbest) + eps) * self.cmaes.parameters.lambda_) + bonus

    def _state(self):
        if self.state_type == StateType.PSB:
            return np.array([self.cmaes.parameters.lambda_, self.cmaes.parameters.sigma, self.cmaes._fitness_func.state.evaluations/self.cmaes.parameters.budget])

    def _update(self, should_continue: bool = True):
        """
        Refresh cached statistics after a CMA-ES generation.

        Parameters
        -
        should_continue : bool
            Return value of cmaes.step(). False means the budget is exhausted.
        """
        self.last_fbest = self.fbest

        self.fbest = self.cmaes._fitness_func.state.current_best_internal.y

        # terminated = budget gone  OR  any modcma convergence criterion fired
        self.terminated = not should_continue or any(self.cmaes.parameters.termination_criteria.values())
        # truncated = optimum found within the BBOB tolerance
        self.truncated = self.cmaes._fitness_func.state.optimum_found