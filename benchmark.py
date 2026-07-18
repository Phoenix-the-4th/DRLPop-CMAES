from time import time
start = time()
from ioh import Experiment, ProblemClass, logger
from cmaes import CMAES
from controllers import IPOP, DefaultPop
import numpy as np

np.random.seed(42)
expno = 4

cmaes = CMAES(DefaultPop())
e = Experiment(
    algorithm = cmaes,
    fids = range(1, 25),
    iids = range(15),
    dims = [2, 3, 5, 10, 20, 40],
    reps = 1,
    problem_class = ProblemClass.BBOB,
    njobs = -1,
    logged = True,
    logger_triggers = [logger.trigger.ON_IMPROVEMENT],
    logger_additional_properties = [],
    output_directory = 'experiments',
    folder_name = f"experiment {expno}",
    algorithm_name = "Default",
    algorithm_info = "",
    zip_output = False,
    run_attributes=["success"])
e.run()

ipop = CMAES(IPOP())
e = Experiment(
    algorithm = ipop,
    fids = range(1, 25),
    iids = range(15),
    dims = [2, 3, 5, 10, 20, 40],
    reps = 1,
    problem_class = ProblemClass.BBOB,
    njobs = -1,
    logged = True,
    logger_triggers = [logger.trigger.ON_IMPROVEMENT],
    logger_additional_properties = [],
    output_directory = 'experiments',
    folder_name = f"experiment {expno}",
    algorithm_name = "IPOP",
    algorithm_info = "",
    zip_output = False,
    run_attributes=["success"])
e.run()

print(time() - start)