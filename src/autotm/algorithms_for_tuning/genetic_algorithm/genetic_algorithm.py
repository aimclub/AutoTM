#!/usr/bin/env python3
import os
import logging
import sys
from logging import config
import warnings
import yaml
from yaml import Loader

import click
import uuid

from autotm.algorithms_for_tuning.genetic_algorithm.ga import GA
from autotm.algorithms_for_tuning.utils import make_log_config_dict

warnings.filterwarnings("ignore")

logger = logging.getLogger("GA")

if "FITNESS_CONFIG_PATH" in os.environ:
    filepath = os.environ["FITNESS_CONFIG_PATH"]
else:
    filepath = "config.yaml"

with open(filepath, "r") as file:
    config = yaml.load(file, Loader=Loader)

glob_algo_params = config["gaAlgoParams"]
NUM_FITNESS_EVALUATIONS = glob_algo_params['numEvals']

NUM_FITNESS_EVALUATIONS = 150


@click.command(context_settings=dict(allow_extra_args=True))
@click.option('--dataset', required=True, help='dataset name in the config')
@click.option('--num-individuals', default=11, help='number of individuals in generation')
@click.option('--num-iterations', default=400, help='number of iterations to make')
@click.option('--num-fitness-evaluations', required=False, type=int, default=None,
              help='Max number of possible fitness estimations. This setting may lead to premature algorithm stopping '
                   'even if there is more generations to go')
@click.option('--mutation-type', default="combined",
              help='mutation type can have value from (mutation_one_param, combined, psm, positioning_mutation)')
@click.option('--crossover-type', default="blend_crossover",
              help='crossover type can have value from (crossover_pmx, crossover_one_point, blend_crossover)')
@click.option('--selection-type', default="fitness_prop",
              help='selection type can have value from (fitness_prop, rank_based)')
@click.option('--elem-cross-prob', default=None, help='crossover probability')
@click.option('--cross-alpha', default=None, help='alpha for blend crossover')
@click.option('--best-proc', default=0.4, help='number of best parents to propagate')
@click.option('--log-file', default="/var/log/tm-alg.log",
              help='a log file to write logs of the algorithm execution to')
@click.option('--exp-id', required=True, type=int, help='mlflow experiment id')
@click.option('--topic-count', required=False, type=int, help='desired count of MAIN topics')
@click.option('--tag', required=False, type=str, help='desired count of MAIN topics')
@click.option('--surrogate-name', required=False, type=str, help='surrogate name')
@click.option('--gpr-kernel', required=False, type=str, help='kernel name for gpr')
@click.option('--gpr-alpha', required=False, type=float, help='alpha for gpr')
@click.option('--gpr-normalize-y', required=False, type=float, help='y normalization for gpr')
def run_algorithm(dataset,
                  num_individuals,
                  num_iterations,
                  num_fitness_evaluations,
                  mutation_type, crossover_type, selection_type,
                  elem_cross_prob, cross_alpha,
                  best_proc, log_file, exp_id, topic_count, tag,
                  surrogate_name, gpr_kernel, gpr_alpha, gpr_normalize_y):
    logger.debug(f"Command line: {sys.argv}")

    run_uid = str(uuid.uuid4())
    tag = tag if tag is not None else str(run_uid)
    logging_config = make_log_config_dict(filename=log_file, uid=run_uid)
    logging.config.dictConfig(logging_config)

    logger.info(f"Starting a new run of algorithm. Args: {sys.argv[1:]}")

    if elem_cross_prob is not None:
        elem_cross_prob = float(elem_cross_prob)

    if cross_alpha is not None:
        cross_alpha = float(cross_alpha)

    g = GA(dataset=dataset,
           num_individuals=num_individuals,
           num_iterations=num_iterations,
           mutation_type=mutation_type,
           crossover_type=crossover_type,
           selection_type=selection_type,
           elem_cross_prob=elem_cross_prob,
           num_fitness_evaluations=num_fitness_evaluations,
           best_proc=best_proc,
           alpha=cross_alpha,
           exp_id=exp_id,
           topic_count=topic_count,
           tag=tag,
           surrogate_name=surrogate_name,
           gpr_kernel=gpr_kernel,
           gpr_alpha=gpr_alpha,
           normalize_y=gpr_normalize_y
           )
    best_value = g.run(verbose=True)
    print(best_value * (-1))



if __name__ == "__main__":
    run_algorithm()
