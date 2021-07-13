#!/usr/bin/env python3

import os
import logging
import click
import numpy as np
from scipy.optimize import differential_evolution
import warnings
import yaml

warnings.filterwarnings("ignore")

logger = logging.getLogger("DE")

# getting config vars
if "FITNESS_CONFIG_PATH" in os.environ:
    filepath = os.environ["FITNESS_CONFIG_PATH"]
else:
    filepath = "../../algorithms_for_tuning/genetic_algorithm/config.yaml"

with open(filepath, "r") as file:
    config = yaml.load(file, Loader=Loader)

if not config['testMode']:
    from kube_fitness.tasks import make_celery_app as prepare_fitness_estimator
    from kube_fitness.tasks import parallel_fitness as estimate_fitness
    from kube_fitness.tasks import log_best_solution
else:
    # from kube_fitness.tm import calculate_fitness_of_individual, TopicModelFactory
    from tqdm import tqdm


    def prepare_fitness_estimator():
        pass


    def estimate_fitness(population: List[IndividualDTO],
                         use_tqdm: bool = False,
                         tqdm_check_period: int = 2) -> List[IndividualDTO]:
        results = []

        tqdm_out = TqdmToLogger(logger, level=logging.INFO)
        for p in tqdm(population, file=tqdm_out):
            individual = copy.deepcopy(p)
            individual.fitness_value = random.random()
            results.append(individual)

        return results


    def log_best_solution(individual: IndividualDTO):
        pass

NUM_FITNESS_EVALUATIONS = config['globalAlgoParams']['numEvals']

def type_check(res):
    res = list(res)
    for i in [1, 4, 7, 10, 11]:
        res[i] = int(res[i])
    return res


@click.command(context_settings=dict(allow_extra_args=True))
@click.option('--dataset', help='dataset name in the config')
@click.option('--strategy', default='best1bin', help='strategy of the algorithm')  # colony size
@click.option('--maxiter', default=None, help='maximum number of generations to evolve')
@click.option('--popsize', default=10, help='population size')
@click.option('--tol', default=None, help='relative tolerance for convergence')
@click.option('--mutation', default=None, help='mutation constant')
@click.option('--recombination', default=None, help='recombinaiton constant')
@click.option('--init', default=None, help='type of population initialization')
@click.option('--atol', default=None, help='absolute tolerance for convergence')
@click.option('--log-file', default="/var/log/tm-alg.log",
              help='a log file to write logs of the algorithm execution to')
def run_algorithm():

    raise NotImplementedError


def BigartmOptimizer(x):
    params = x
    print('PARAMS: ', params)

    t = Topic_model(experiments_path, S=S)

    params = type_check(params)
    t.init_model(params)
    t.train()

    try:
        scores = get_last_avg_vals(t.model, S)
        print(scores)
    except:
        print('NO SCORES SHALL PASS!')

    # Additional penalty for creating unmeaningfull main topics
    #     theta = t.model.get_theta()
    #     theta_trans = theta.T[['main{}'.format(i) for i in range(S)]]
    #     theta_trans_vals = theta_trans.max(axis=1)
    #     fine_elems_coeff = len(np.where(np.array(theta_trans_vals)>=0.2)[0])/theta_trans.shape[0]
    try:
        fitness = t.get_avg_coherence_score(for_individ_fitness=True)  # *fine_elems_coeff
    except:
        fitness = 0
    if np.isnan(fitness):
        fitness = 0
    print()
    print('CURRENT FITNESS: {}'.format(fitness))
    print()
    return -fitness


if __name__ == "__main__":
    run_algorithm()
