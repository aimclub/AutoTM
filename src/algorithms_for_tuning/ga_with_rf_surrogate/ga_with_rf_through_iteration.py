#!/usr/bin/env python3

import os
import sys
import click
import uuid
import logging
import warnings
import yaml
from yaml import Loader
import copy
import random
from typing import List

from algorithms_for_tuning.genetic_algorithm.mutation import mutation
from algorithms_for_tuning.genetic_algorithm.crossover import crossover
from algorithms_for_tuning.genetic_algorithm.selection import selection
from algorithms_for_tuning.utils import make_log_config_dict

warnings.filterwarnings("ignore")

from kube_fitness.tasks import IndividualDTO, TqdmToLogger

logger = logging.getLogger("GA")

# getting config vars
if "FITNESS_CONFIG_PATH" in os.environ:
    filepath = os.environ["FITNESS_CONFIG_PATH"]
else:
    filepath = "../../algorithms_for_tuning/ga_with_rf_surrogate/config.yaml"

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

# TODO: check types correctness
NUM_FITNESS_EVALUATIONS = config['globalAlgoParams']['numEvals']
NUM_INDIVIDUALS = config['globalAlgoParams']['numIndividuals']
MUTATION_TYPE = config['globalAlgoParams']['mutationType']
CROSSOVER_TYPE = config['globalAlgoParams']['crossoverType']
SELECTION_TYPE = config['globalAlgoParams']['selectionType']
CROSS_ALPHA = config['globalAlgoParams']['crossAlpha']
BEST_PROC = config['globalAlgoParams']['bestProc']
ELEM_CROSS_PROB = config['globalAlgoParams']['elemCrossProb']


@click.command(context_settings=dict(allow_extra_args=True))
@click.option('--dataset', required=True, help='dataset name in the config')

@click.option('--log-file', default="/var/log/tm-alg.log",
              help='a log file to write logs of the algorithm execution to')
@click.option('--exp-id', required=True, type=int, help='mlflow experiment id')
def run_algorithm(dataset,
                  num_individuals,
                  mutation_type, crossover_type, selection_type,
                  elem_cross_prob, cross_alpha,
                  best_proc, log_file, exp_id):
    run_uid = uuid.uuid4() if not config['testMode'] else None
    logging_config = make_log_config_dict(filename=log_file, uid=run_uid)
    logging.config.dictConfig(logging_config)

    logger.info(f"Starting a new run of algorithm. Args: {sys.argv[1:]}")

    if elem_cross_prob is not None:
        elem_cross_prob = float(elem_cross_prob)

    if cross_alpha is not None:
        cross_alpha = float(cross_alpha)

    g = GA(dataset=dataset,
           num_individuals=NUM_INDIVIDUALS,
           num_iterations=400,
           mutation_type=MUTATION_TYPE,
           crossover_type=CROSSOVER_TYPE,
           selection_type=SELECTION_TYPE,
           elem_cross_prob=ELEM_CROSS_PROB,
           num_fitness_evaluations=NUM_FITNESS_EVALUATIONS,
           best_proc=BEST_PROC,
           alpha=CROSS_ALPHA,
           exp_id=exp_id)
    best_value = g.run(verbose=True)
    print(best_value * (-1))


# class GA