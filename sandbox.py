import warnings
warnings.filterwarnings("ignore")

from ioh import problem, get_problem, ProblemClass, suite, ProblemType
from ioh import logger

# s = suite.BBOB(list(problem.BBOB.problems.keys()), instances=list(range(20)))
# print(s.problem_ids)
# print(s.instances)
# print(s.dimensions)


p = get_problem(1, 0, 2, ProblemClass.BBOB)
l = logger.Analyzer(triggers=[logger.trigger.ALWAYS], folder_name="results/tempres")
p.attach_logger(l)
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



from cmaes import CMAES
from controllers import IPOP, Default
ipop = IPOP()
cma = CMAES(ipop)
a = cma.run_verb(p, "BBOB")
p.reset()
cpop = Default()
cma = CMAES(cpop)
a = cma.run_verb(p, "BBOB")
p.reset()
l.close()



# from ioh import Experiment
# e = Experiment(cma, [1, 2], [0, 1, 2, 3, 4], [2, 5])
# e.run()


















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