#!/usr/bin/env python3
import copy
import logging
import os
import warnings
from typing import List, Optional

# from kube_fitness.tasks import make_celery_app, parallel_fitness, IndividualDTO
# from kube_fitness.tasks import IndividualDTO, TqdmToLogger
import click
import yaml
from numpy import random
from yaml import Loader
from kube_fitness.tasks import IndividualDTO, TqdmToLogger

warnings.filterwarnings("ignore")

logger = logging.getLogger("gray_wolf")

# TODO: getting config vars
if "FITNESS_CONFIG_PATH" in os.environ:
    filepath = os.environ["FITNESS_CONFIG_PATH"]
else:
    filepath = "../../algorithms_for_tuning/gray_wolf_algorithm/config.yaml"

with open(filepath, "r") as file:
    config = yaml.load(file, Loader=Loader)

if not config['testMode']:
    from kube_fitness.tasks import make_celery_app as prepare_fitness_estimator
else:
    # from kube_fitness.tm import calculate_fitness_of_individual, TopicModelFactory
    from tqdm import tqdm


    def prepare_fitness_estimator():
        pass


    def estimate_fitness(population: List[IndividualDTO],
                         _: bool = False,
                         __: int = 2) -> List[IndividualDTO]:
        results = []

        tqdm_out = TqdmToLogger(logger, level=logging.INFO)
        for p in tqdm(population, file=tqdm_out):
            individual = copy.deepcopy(p)
            individual.fitness_value = random.random()
            results.append(individual)

        return results


    def log_best_solution(individual: IndividualDTO, alg_args: Optional[str]):
        pass


# TODO: get fitness eval
# NUM_FITNESS_EVALUATIONS = config['abcAlgoParams']['numEvals']
# PROBLEM_DIM = config['abcAlgoParams']['problemDim']
@click.command(context_settings=dict(allow_extra_args=True))
@click.option('--dataset', required=True, help='dataset name in the config')
@click.option('--num-individuals', default=10, help='colony size')  # colony size
@click.option('--max-num-trials', default=5, help='maximum number of source trials')
@click.option('--init-method', default='latin_hypercube',
              help='method of population initialization (latin hypercube or random)')
@click.option('--log-file', default="/var/log/tm-alg.log",
              help='a log file to write logs of the algorithm execution to')
@click.option('--exp-id', required=True, type=int, help='mlflow experiment id')
class Wolf:
    def __init__(self):
        raise NotImplementedError()


class GrayWolf:
    def __init__(self):
        pass

    def run(self):
        prepare_fitness_estimator()

        raise NotImplementedError()
