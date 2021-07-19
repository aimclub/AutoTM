#!/usr/bin/env python3
import logging
import sys
from logging import config
import warnings

import click
import uuid

from ga import GA
from algorithms_for_tuning.utils import make_log_config_dict

warnings.filterwarnings("ignore")

logger = logging.getLogger("GA")

NUM_FITNESS_EVALUATIONS = config['globalAlgoParams']['numEvals']


@click.command(context_settings=dict(allow_extra_args=True))
@click.option('--dataset', required=True, help='dataset name in the config')
@click.option('--num-individuals', default=10, help='number of individuals in generation')
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
def run_algorithm(dataset,
                  num_individuals,
                  mutation_type, crossover_type, selection_type,
                  elem_cross_prob, cross_alpha,
                  best_proc, log_file, exp_id):
    run_uid = str(uuid.uuid4())
    logging_config = make_log_config_dict(filename=log_file, uid=run_uid)
    logging.config.dictConfig(logging_config)

    logger.info(f"Starting a new run of algorithm. Args: {sys.argv[1:]}")

    if elem_cross_prob is not None:
        elem_cross_prob = float(elem_cross_prob)

    if cross_alpha is not None:
        cross_alpha = float(cross_alpha)

    g = GA(dataset=dataset,
           num_individuals=num_individuals,
           num_iterations=400,
           mutation_type=mutation_type,
           crossover_type=crossover_type,
           selection_type=selection_type,
           elem_cross_prob=elem_cross_prob,
           num_fitness_evaluations=NUM_FITNESS_EVALUATIONS,
           best_proc=best_proc,
           alpha=cross_alpha,
           exp_id=exp_id)
    best_value = g.run(verbose=True, alg_args=" ".join(sys.argv))
    print(best_value * (-1))


if __name__ == "__main__":
    run_algorithm()
