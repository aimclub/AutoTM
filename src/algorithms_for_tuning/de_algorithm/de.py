#!/usr/bin/env python3

import os
import logging
from typing import List, Optional
import click
import uuid
import numpy as np
from scipy.optimize import differential_evolution, OptimizeResult
import warnings
import yaml
from yaml import Loader
import copy
import random

from algorithms_for_tuning.utils import make_log_config_dict
from kube_fitness.tasks import IndividualDTO, TqdmToLogger

warnings.filterwarnings("ignore")

logger = logging.getLogger("DE")

# getting config vars
if "FITNESS_CONFIG_PATH" in os.environ:
    filepath = os.environ["FITNESS_CONFIG_PATH"]
else:
    filepath = "../../algorithms_for_tuning/de_algorithm/config.yaml"

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

DATASET_NAME = None
EXP_ID = None
ALG_ID = 'de'
BEST_SOLUTION = None

HIGH_DECOR = 1e5
LOW_DECOR = 0
HIGH_N = 8
LOW_N = 0
HIGH_SM = 1e2
LOW_SM = 1e-3
HIGH_SP = -1e-3
LOW_SP = -1e2

BOUNDS = [
    (LOW_DECOR, HIGH_DECOR),
    (LOW_N, HIGH_N),
    (LOW_SM, HIGH_SM),
    (LOW_SM, HIGH_SM),
    (LOW_N, HIGH_N),
    (LOW_SP, HIGH_SP),
    (LOW_SP, HIGH_SP),
    (LOW_N, HIGH_N),
    (LOW_SP, HIGH_SP),
    (LOW_SP, HIGH_SP),
    (LOW_N, HIGH_N),
    (LOW_N, HIGH_N),
    (LOW_DECOR, HIGH_DECOR)
]


class BigartmFitness:

    def __init__(self, dataset: str, exp_id: Optional[int] = None):
        self.dataset = dataset
        self.exp_id = exp_id
        # self.best_solution: Optional[IndividualDTO] = None

    def make_individ(self, x):
        params = [float(i) for i in x]
        params = params[:-1] + [0.0, 0.0, 0.0] + [params[-1]]
        return IndividualDTO(
            id=str(uuid.uuid4()),
            dataset=self.dataset,
            params=params,
            exp_id=self.exp_id,
            alg_id=ALG_ID
        )

    def __call__(self, x):
        population = [self.make_individ(x)]

        population = estimate_fitness(population)
        individ = population[0]

        # if self.best_solution is None or individ.fitness_value > self.best_solution.fitness_value:
        #     self.best_solution = copy.deepcopy(individ)

        return -1 * individ.fitness_value


@click.command(context_settings=dict(allow_extra_args=True))
@click.option('--dataset', required=True, type=str, help='dataset name in the config')
@click.option('--strategy', type=str, default='best1bin', help='strategy of the algorithm')  # colony size
@click.option('--popsize', type=int, default=10, help='population size')
@click.option('--tol', type=float, default=None, help='relative tolerance for convergence')
@click.option('--mutation', type=float, default=None, help='mutation constant')
@click.option('--recombination', type=float, default=None, help='recombinaiton constant')
@click.option('--init', type=str, default=None, help='type of population initialization')
@click.option('--atol', type=float, default=None, help='absolute tolerance for convergence')
@click.option('--log-file', type=str, default="/var/log/tm-alg.log",
              help='a log file to write logs of the algorithm execution to')
@click.option('--exp-id', required=True, type=int, help='mlflow experiment id')
def run_algorithm(dataset, strategy, popsize,
                  tol, mutation, recombination,
                  init, atol, log_file, exp_id):
    global DATASET_NAME
    DATASET_NAME = dataset
    global EXP_ID
    EXP_ID = exp_id
    maxiter = int(np.ceil(NUM_FITNESS_EVALUATIONS / popsize))
    prepare_fitness_estimator()
    fitness = BigartmFitness(dataset, exp_id)
    res_fitness: OptimizeResult = differential_evolution(
        fitness,
        BOUNDS,
        strategy=strategy,
        maxiter=maxiter,
        popsize=popsize,
        tol=tol,
        mutation=mutation,
        recombination=recombination,
        init=init,
        atol=atol
    )

    if not res_fitness.success:
        raise Exception(
            f"The DE run has failed. "
            f"Number of performed iterations(nit): {res_fitness.nit if 'nit' in res_fitness else 'unknown'}. "
            f"Number of fitness evaluations (nfev): {res_fitness.nfev if 'nfev' in res_fitness else 'unknown'} "
            f"Message: {res_fitness.message}."
        )

    best_solution = fitness.make_individ(res_fitness.x)
    log_best_solution(best_solution)
    print(res_fitness.fun * (-1))


def type_check(res):
    res = list(res)
    for i in [1, 4, 7, 10, 11]:
        res[i] = int(res[i])
    return res


if __name__ == "__main__":
    run_algorithm()
