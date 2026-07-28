import warnings
warnings.filterwarnings("ignore")

from ioh import problem, get_problem, ProblemClass, suite, ProblemType
from ioh import logger

# s = suite.BBOB(list(problem.BBOB.problems.keys()), instances=list(range(20)))
# print(s.problem_ids)
# print(s.instances)
# print(s.dimensions)


# p = get_problem(1, 0, 2, ProblemClass.BBOB)
# l = logger.Analyzer(triggers=[logger.trigger.ALWAYS], additional_properties=[logger.property.CURRENTY, logger.property.TRANSFORMEDY], folder_name="results/tempres")
# p.attach_logger(l)
# help(logger.Analyzer)
# print(isinstance(p, problem.BBOB))
# print(type(p), isinstance(p, ProblemType))
# print(p.meta_data.problem_id)
# print(p.meta_data.name)
# print(p.meta_data.instance)
# print(p.meta_data.n_variables)
# print(p.meta_data.optimization_type)



# print(p.state)
# print(p.optimum)
# print(p([0, 0]))
# print(p.state)
# print(p([1, 1]))
# print(p.state)
# print(p([-1, -1]))
# print(p.state)
# print(p(p.optimum.x + [0.0001, 0]))
# print(p.state)
# p.reset()
# l.close()



# from cmaes import CMAES
# from controllers import IPOP, DefaultPop
import numpy as np
np.random.seed(0)
# ipop = IPOP()
# cma = CMAES(ipop)
# a = cma.run_restart(p)
# p.reset()
# cpop = DefaultPop()
# cma = CMAES(controller= cpop)
# a = cma.run(p)
# p.reset()
# l.close()





















# ------------------------------------ DACBench

# from dacbench.benchmarks import CMAESBenchmark
# from ioh import get_problem, ProblemClass

# bench = CMAESBenchmark()
# bench.set_seed(0)
# [print(k, getattr(bench.config, k)) for k in bench.config]
# print()
# print()

# env = bench.get_environment()


# env.reset()
# print(env.get_state(), env.get_reward())
# print()
# print(env.step({}))
# print()
# print(env.get_state(), env.get_reward())


# print(env.instance)

# from cmaes import CMAEnv
# e = CMAEnv()
# print(e)
# print(e.reset())
# print(e.last_fval,
#     e.last_fbest,
#     e.fval,
#     e.fbest,
#     e.terminated,
#     e.completed)

f1 = r"experiments\experiment 4"
f2 = r"experiment 4-1"

import os
import json
al, bl = [], []
for fname in os.listdir(f1):
    if fname.endswith('.json'):
        d1 = json.load(open(os.path.join(f1, fname), 'r'))
        d2 = json.load(open(os.path.join(f2, fname), 'r'))
        a = ([(scenario['dimension'], sum([run['success'] for run in scenario['runs']])) for scenario in d1['scenarios']])
        al.append(a)
        b = ([(scenario['dimension'], sum([run['success'] for run in scenario['runs']])) for scenario in d2['scenarios']])
        bl.append(b)
# print(al)
d = {i: [0, 0] for i in [2, 3, 5, 10, 20, 40]}  # dim: [ipop, drl successes]
for l in al:
    for t in l:
        # print(d[t[0]], t[1])
        d[t[0]][0] += t[1]
for l in bl:
    for t in l:
        d[t[0]][1] += t[1]

print(d)
print([100*a/b for a, b in d.values()])