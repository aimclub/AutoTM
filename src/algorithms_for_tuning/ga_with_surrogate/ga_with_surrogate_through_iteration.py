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
from algorithms_for_tuning.genetic_algorithm.ga import GA
from algorithms_for_tuning.utils import make_log_config_dict
from kube_fitness.tasks import IndividualDTO, TqdmToLogger

warnings.filterwarnings("ignore")

logger = logging.getLogger("GA_surrogate")

# getting config vars
if "FITNESS_CONFIG_PATH" in os.environ:
    filepath = os.environ["FITNESS_CONFIG_PATH"]
else:
    filepath = "../../algorithms_for_tuning/ga_with_surrogate/config.yaml"

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

# TODO: check types correctness & None processing
NUM_FITNESS_EVALUATIONS = config['globalAlgoParams']['numEvals']
NUM_INDIVIDUALS = config['globalAlgoParams']['numIndividuals']
MUTATION_TYPE = config['globalAlgoParams']['mutationType']
CROSSOVER_TYPE = config['globalAlgoParams']['crossoverType']
SELECTION_TYPE = config['globalAlgoParams']['selectionType']
CROSS_ALPHA = config['globalAlgoParams']['crossAlpha']
BEST_PROC = config['globalAlgoParams']['bestProc']
ELEM_CROSS_PROB = config['globalAlgoParams']['elemCrossProb']

if isinstance(ELEM_CROSS_PROB, str):
    ELEM_CROSS_PROB = None
else:
    ELEM_CROSS_PROB = float(ELEM_CROSS_PROB)


@click.command(context_settings=dict(allow_extra_args=True))
@click.option('--dataset', required=True, help='dataset name in the config')
@click.option('--log-file', default="/var/log/tm-alg.log",
              help='a log file to write logs of the algorithm execution to')
@click.option('--exp-id', required=True, type=int, help='mlflow experiment id')
@click.option('--surrogate-name', default=None, type=str,
              help="Enter surrogate name if you want to start calculations with surrogate")
@click.option('--rf-n-estimators', type=int)
@click.option('--rf-criterion', type=str)
@click.option('--rf-max-depth', type=int)
@click.option('--rf-min-samples-split', type=float)
@click.option('--rf-min-samples-leaf', type=float)
@click.option('--rf-min-weight-fraction-leaf', type=float)
@click.option('--rf-max-features', type=float)
@click.option('--rf-oob-score', type=bool)
@click.option('--rf-n-jobs', type=int)
def run_algorithm(dataset, log_file, exp_id, surrogate_name,
                  rf_n_estimators, rf_criterion, rf_max_depth, rf_min_samples_split,
                  rf_min_samples_leaf, rf_min_weight_fraction_leaf, rf_max_features, rf_oob_score, rf_n_jobs,

                  ):
    run_uid = uuid.uuid4() if not config['testMode'] else None
    logging_config = make_log_config_dict(filename=log_file, uid=run_uid)
    logging.config.dictConfig(logging_config)

    kwargs = dict()
    if surrogate_name == 'random-forest-regressor':
        kwargs = {
                  'n_estimators': rf_n_estimators,
                  'criterion': rf_criterion,
                  'max_depth': rf_max_depth,
                  'min_samples_split': rf_min_samples_split,
                  'min_samples_leaf': rf_min_samples_leaf,
                  'min_weight_fraction_leaf': rf_min_weight_fraction_leaf,
                  'max_features': rf_max_features,
                  'oob_score': rf_oob_score,
                  'n_jobs': rf_n_jobs,

                  }
    elif surrogate_name == 'mlp-regressor':
        raise NotImplementedError

    logger.info(f"Starting a new run of algorithm. Args: {sys.argv[1:]}")

    if CROSS_ALPHA is not None:
        cross_alpha = float(CROSS_ALPHA)

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
           exp_id=exp_id,
           **kwargs)
    best_value = g.run(verbose=True)
    print(best_value * (-1))

if __name__=="__main__":
    run_algorithm()