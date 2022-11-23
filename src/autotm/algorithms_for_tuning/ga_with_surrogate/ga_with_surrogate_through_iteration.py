#!/usr/bin/env python3

import logging
import os
import sys
import uuid
import warnings
from logging import config

import click
import yaml
from yaml import Loader

warnings.simplefilter("ignore")

from algorithms_for_tuning.genetic_algorithm.ga import GA
from algorithms_for_tuning.utils import make_log_config_dict, log_any_error

warnings.filterwarnings("ignore")

logger = logging.getLogger("GA_surrogate")

# getting config vars
if "FITNESS_CONFIG_PATH" in os.environ:
    filepath = os.environ["FITNESS_CONFIG_PATH"]
else:
    filepath = "config.yaml"

with open(filepath, "r") as file:
    config = yaml.load(file, Loader=Loader)

glob_algo_params = config["gaWithSurrogateAlgoParams"]
NUM_FITNESS_EVALUATIONS = glob_algo_params['numEvals']
NUM_INDIVIDUALS = glob_algo_params['numIndividuals']
MUTATION_TYPE = glob_algo_params['mutationType']
CROSSOVER_TYPE = glob_algo_params['crossoverType']
SELECTION_TYPE = glob_algo_params['selectionType']
CROSS_ALPHA = glob_algo_params['crossAlpha']
BEST_PROC = glob_algo_params['bestProc']
ELEM_CROSS_PROB = glob_algo_params['elemCrossProb']

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
@click.option('--rf-max-depth', type=str)
@click.option('--rf-min-samples-split', type=float)
@click.option('--rf-min-samples-leaf', type=float)
@click.option('--rf-min-weight-fraction-leaf', type=float)
@click.option('--rf-max-features', type=str)
@click.option('--rf-oob-score', type=bool)
@click.option('--rf-n-jobs', type=int)
@click.option('--br-n-estimators', type=int)
@click.option('--br-n-jobs', type=int)
@click.option('--mlp-activation', type=str)
@click.option('--mlp-hidden-layer-sizes_1', type=int)
@click.option('--mlp-hidden-layer-sizes_2', type=int)
@click.option('--mlp-hidden-layer-sizes_3', type=int)
@click.option('--mlp-solver', type=str)
@click.option('--mlp-alpha', type=float)
@click.option('--mlp-learning-rate', type=str)
@click.option('--mlp-max-iter', type=int)
@click.option('--mlp-momentum', type=float)
@click.option('--mlp-early-stopping', type=bool)
@click.option('--gpr-kernel', type=str)
@click.option('--gpr-alpha', type=float)
@click.option('--gpr-normalize-y', type=bool)
@click.option('--dt-criterion', type=str)
@click.option('--dt-splitter', type=str)
@click.option('--dt-max-depth', type=int)
@click.option('--dt-min-samples-split', type=float)
@click.option('--dt-min-samples-leaf', type=float)
@click.option('--dt-max-features', type=str)
@click.option('--dt-ccp-alpha', type=float)
@click.option('--svr-kernel', type=str)
@click.option('--svr-degree', type=int, default=3)
@click.option('--svr-gamma', type=str)
@click.option('--svr-coef0', type=float, default=0.0)
@click.option('--svr-c', type=float)
@click.option('--svr-epsilon', type=float)
@click.option('--svr-max-iter', type=int)
@click.option('--calc-scheme', type=str, default='type1')
def run_algorithm(dataset, log_file, exp_id, surrogate_name,
                  rf_n_estimators, rf_criterion, rf_max_depth, rf_min_samples_split,
                  rf_min_samples_leaf, rf_min_weight_fraction_leaf, rf_max_features, rf_oob_score, rf_n_jobs,
                  br_n_estimators, br_n_jobs, mlp_activation, mlp_hidden_layer_sizes_1, mlp_hidden_layer_sizes_2,
                  mlp_hidden_layer_sizes_3, mlp_solver, mlp_alpha, mlp_learning_rate, mlp_max_iter,
                  mlp_momentum, mlp_early_stopping,
                  gpr_kernel, gpr_alpha, gpr_normalize_y,
                  dt_criterion, dt_splitter, dt_max_depth, dt_min_samples_split, dt_min_samples_leaf,
                  dt_max_features, dt_ccp_alpha,
                  svr_kernel, svr_degree, svr_gamma, svr_coef0, svr_c, svr_epsilon, svr_max_iter,
                  calc_scheme,
                  ):
    run_uid = uuid.uuid4() if not config['testMode'] else None
    logging_config = make_log_config_dict(filename=log_file, uid=run_uid)
    logging.config.dictConfig(logging_config)

    kwargs = dict()
    if surrogate_name == 'random-forest-regressor':
        if rf_max_depth == 'None':
            rf_max_depth = None
        else:
            rf_max_depth = int(rf_max_depth)
        if rf_min_samples_split == 2:
            rf_min_samples_split = int(rf_min_samples_split)
        if rf_min_samples_leaf == 1:
            rf_min_samples_leaf = int(rf_min_samples_leaf)
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
        kwargs = {k: v for k, v in kwargs.items() if v is not None}
    elif surrogate_name == 'mlp-regressor':
        kwargs = {
            'br_n_estimators': br_n_estimators,
            'n_jobs': br_n_jobs,
            'activation': mlp_activation,
            'hidden_layer_sizes': (mlp_hidden_layer_sizes_1, mlp_hidden_layer_sizes_2, mlp_hidden_layer_sizes_3),
            'solver': mlp_solver,
            'mlp_alpha': mlp_alpha,
            'learning_rate': mlp_learning_rate,
            'max_iter': mlp_max_iter,
            'momentum': mlp_momentum,
            'early_stopping': mlp_early_stopping,
        }
        kwargs = {k: v for k, v in kwargs.items() if v is not None}
        if 'n_jobs' not in kwargs:
            kwargs['n_jobs'] = None
    elif surrogate_name == 'GPR':
        kwargs = {
            'gpr_kernel': gpr_kernel,
            'gpr_alpha': gpr_alpha,
            'normalize_y': gpr_normalize_y,
        }
        kwargs = {k: v for k, v in kwargs.items() if v is not None}
    elif surrogate_name == 'decision-tree-regressor':
        if dt_min_samples_split == 2:
            dt_min_samples_split = int(dt_min_samples_split)
        if dt_min_samples_leaf == 1:
            dt_min_samples_leaf = int(dt_min_samples_leaf)
        kwargs = {
            'criterion': dt_criterion,
            'splitter': dt_splitter,
            'max_depth': dt_max_depth,
            'min_samples_split': dt_min_samples_split,
            'min_samples_leaf': dt_min_samples_leaf,
            'max_features': dt_max_features,
            'ccp_alpha': dt_ccp_alpha,
        }
        kwargs = {k: v for k, v in kwargs.items() if v is not None}
    elif surrogate_name == 'SVR':
        kwargs = {
            'kernel': svr_kernel,
            'degree': svr_degree,
            'gamma': svr_gamma,
            'coef0': svr_coef0,
            'C': svr_c,
            'epsilon': svr_epsilon,
            'max_iter': svr_max_iter,
        }

    logger.info(f"Starting a new run of algorithm. Args: {sys.argv[1:]}")

    cross_alpha = float(CROSS_ALPHA) if CROSS_ALPHA else None

    g = GA(dataset=dataset,
           num_individuals=NUM_INDIVIDUALS,
           num_iterations=400,
           mutation_type=MUTATION_TYPE,
           crossover_type=CROSSOVER_TYPE,
           selection_type=SELECTION_TYPE,
           elem_cross_prob=ELEM_CROSS_PROB,
           num_fitness_evaluations=NUM_FITNESS_EVALUATIONS,
           best_proc=BEST_PROC,
           alpha=cross_alpha,
           exp_id=exp_id,
           surrogate_name=surrogate_name,
           calc_scheme=calc_scheme,
           **kwargs)
    best_value = g.run(verbose=True)
    print(best_value * (-1))


if __name__ == "__main__":
    with log_any_error():
        run_algorithm()
