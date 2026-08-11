from time import time
start = time()
from ioh import Experiment, ProblemClass, logger
from cmaes import CMAES
from environments import CMAEnv, StateType
from controllers import *
import numpy as np
import os

import random

seed = 1
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.backends.cudnn.deterministic = True


def run_exp_weights(path, outdir, fname, algo_name, algo_info, limit  = 2):
    dpop = DRLPop(path, StateType.PSB.value, -limit, limit)
    cma = CMAES(controller= dpop)
    e = Experiment(
        algorithm = cma,
        fids = range(1, 25),
        iids = range(15),
        dims = [2, 3, 5, 10, 20, 40],
        reps = 1,
        problem_class = ProblemClass.BBOB,
        njobs = -1,
        logged = True,
        logger_triggers = [logger.trigger.ON_IMPROVEMENT],
        logger_additional_properties = [],
        output_directory = outdir,
        folder_name = f"{fname}",
        algorithm_name = algo_name,
        algorithm_info = algo_info,
        zip_output = False,
        run_attributes=["success"])
    e.run()

def run_exp_folder(model_folder, outdir, algo_name, algo_info = ""):
    for f in os.listdir(model_folder):
        for weights in os.listdir(os.path.join(model_folder, f)):
            if weights.endswith('.pt'):
                run_exp_weights(os.path.join(model_folder, f, weights), outdir, f, algo_name, algo_info)


if __name__ == "__main__":
    # example usage
    run_exp_weights('m0', 'r0', 'DRLPop')
    print(time() - start)