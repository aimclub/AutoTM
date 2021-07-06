#!/usr/bin/env python3.6
import logging
import sys
from logging import config

import os
import warnings
from typing import List

import click
import random
import numpy as np
import time
import copy
import operator
import uuid
import gc
import yaml
from yaml import Loader

from mutation import mutation
from crossover import crossover
from selection import selection
from src.algorithms_for_tuning.utils import make_log_config_dict

warnings.filterwarnings("ignore")

from kube_fitness.tasks import IndividualDTO, TqdmToLogger

logger = logging.getLogger("GA")

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


NUM_FITNESS_EVALUATIONS = config['globalAlgoParams']['numEvals']
LOG_FILE_PATH = config['paths']['logFile']
# DATASET = "20newsgroups"


# TODO: add irace default params
@click.command(context_settings=dict(allow_extra_args=True))
@click.option('--dataset', help='dataset name in the config')
@click.option('--num-individuals', default=10, help='number of individuals in generation')
@click.option('--mutation-type', default="combined",
              help='mutation type can have value from (mutation_one_param, combined, psm, positioning_mutation)')
@click.option('--crossover-type', default="blend_crossover",
              help='crossover type can have value from (crossover_pmx, crossover_one_point, blend_crossover)')
@click.option('--selection-type', default="fitness_prop",
              help='selection type can have value from (fitness_prop, rank_based)')
@click.option('--elem-cross-prob', default=None, help='crossover probability')
@click.option('--cross-alpha', default=None, help='alpha for blend crosover')
@click.option('--best-proc', default=0.4, help='number of best parents to propagate')
@click.option('--log-file', default="/var/log/tm-alg.log", help='a log file to write logs of the algorithm execution to')
def run_algorithm(dataset,
                  num_individuals,
                  mutation_type, crossover_type, selection_type,
                  elem_cross_prob, cross_alpha,
                  best_proc, log_file):
    logging_config = make_log_config_dict(filename=log_file)
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
           alpha=cross_alpha)
    best_value = g.run(verbose=True)
    print(best_value * (-1))


class GA:
    def __init__(self, dataset, num_individuals, num_iterations,
                 mutation_type='mutation_one_param', crossover_type='blend_crossover',
                 selection_type='fitness_prop', elem_cross_prob=0.2, num_fitness_evaluations=200,
                 best_proc=0.3, alpha=None,):
        self.dataset = dataset

        if crossover_type == 'blend_crossover':
            self.crossover_children = 1
        else:
            self.crossover_children = 2

        self.num_individuals = num_individuals
        self.num_iterations = num_iterations
        self.mutation = mutation(mutation_type)  # mutation function
        self.crossover = crossover(crossover_type)  # crossover finction
        self.selection = selection(selection_type)  # selection function
        self.elem_cross_prob = elem_cross_prob
        self.alpha = alpha
        self.num_fitness_evaluations = num_fitness_evaluations
        self.best_proc = best_proc

    @staticmethod
    def init_individ(high_decor=1e5,
                     high_n=8, high_spb=1e2,
                     low_spm=-1e2):
        val_decor = np.random.uniform(low=0, high=high_decor, size=1)[0]
        var_n = np.random.randint(low=0, high=high_n, size=5)
        var_sm = np.random.uniform(low=1e-3, high=high_spb, size=2)
        var_sp = np.random.uniform(low=low_spm, high=-1e-3, size=4)
        ext_mutation_prob = np.random.uniform(low=0, high=1, size=1)[0]
        ext_elem_mutation_prob = np.random.uniform(low=0, high=1, size=1)[0]
        ext_mutation_selector = np.random.uniform(low=0, high=1, size=1)[0]
        val_decor_2 = np.random.uniform(low=0, high=high_decor, size=1)[0]
        params = [
            val_decor, var_n[0],
            var_sm[0], var_sm[1], var_n[1],
            var_sp[0], var_sp[1], var_n[2],
            var_sp[2], var_sp[3], var_n[3],
            var_n[4],
            ext_mutation_prob, ext_elem_mutation_prob, ext_mutation_selector,
            val_decor_2
        ]
        params = [float(i) for i in params]
        return params

    def init_population(self):
        list_of_individuals = []
        for i in range(self.num_individuals):
            list_of_individuals.append(IndividualDTO(id=str(uuid.uuid4()),
                                                     dataset=self.dataset,
                                                     params=self.init_individ()))
        population_with_fitness = estimate_fitness(list_of_individuals)
        return population_with_fitness

    def run(self, verbose=False):
        prepare_fitness_estimator()

        evaluations_counter = 0
        ftime = str(int(time.time()))

        os.makedirs(LOG_FILE_PATH, exist_ok=True)

        logger.info(f"Starting experiment: {ftime}")

        logger.info(f"ALGORITHM PARAMS  number of individuals {self.num_individuals}; "
                    f"number of fitness evals {self.num_fitness_evaluations}; "
                    f"crossover prob {self.elem_cross_prob}")

        # population initialization
        population = self.init_population()

        evaluations_counter = self.num_individuals

        logger.info("POPULATION IS CREATED")

        x, y = [], []
        high_fitness = 0
        for ii in range(self.num_iterations):

            new_generation = []

            logger.info(f"ENTERING GENERATION {ii}")

            population.sort(key=operator.attrgetter('fitness_value'), reverse=True)
            pairs_generator = self.selection(population=population,
                                             best_proc=self.best_proc, children_num=self.crossover_children)

            logger.info(f"PAIRS ARE CREATED")

            # Crossover
            for i, j in pairs_generator:

                if i is None:
                    break

                parent_1 = copy.deepcopy(i.params)
                parent_2 = copy.deepcopy(j.params)

                if self.crossover_children == 2:

                    child_1, child_2 = self.crossover(parent_1=parent_1,
                                                      parent_2=parent_2,
                                                      elem_cross_prob=self.elem_cross_prob,
                                                      alpha=self.alpha)

                    new_generation.append(IndividualDTO(id=str(uuid.uuid4()),
                                                        dataset=self.dataset,
                                                        params=child_1))
                    new_generation.append(IndividualDTO(id=str(uuid.uuid4()),
                                                        dataset=self.dataset,
                                                        params=child_2))
                    evaluations_counter += 2
                else:
                    child_1 = self.crossover(parent_1=parent_1,
                                             parent_2=parent_2,
                                             elem_cross_prob=self.elem_cross_prob,
                                             alpha=self.alpha
                                             )
                    new_generation.append(IndividualDTO(id=str(uuid.uuid4()),
                                                        dataset=self.dataset,
                                                        params=child_1))

                    evaluations_counter += 1

            logger.info(f"CURRENT COUNTER: {evaluations_counter}")

            new_generation = estimate_fitness(new_generation)

            new_generation.sort(key=operator.attrgetter('fitness_value'), reverse=True)
            population.sort(key=operator.attrgetter('fitness_value'), reverse=True)

            logger.info("CROSSOVER IS OVER")

            if evaluations_counter >= self.num_fitness_evaluations:
                bparams = ''.join([str(i) for i in population[0].params])
                logger.info(f"TERMINATION IS TRIGGERED."
                            f"THE BEST FITNESS {population[0].fitness_value}."
                            f"THE BEST PARAMS {bparams}.")
                return population[0].fitness_value

            del pairs_generator
            gc.collect()

            population_params = [copy.deepcopy(individ.params) for individ in population]

            the_best_guy_params = copy.deepcopy(population[0].params)
            new_generation = [individ for individ in new_generation if individ.params != the_best_guy_params]

            new_generation_n = min((self.num_individuals - int(np.ceil(self.num_individuals * self.best_proc))), len(new_generation))
            old_generation_n = self.num_individuals - new_generation_n

            population = population[:old_generation_n] + new_generation[:new_generation_n]

            try:
                del new_generation
            except:
                pass

            gc.collect()

            population.sort(key=operator.attrgetter('fitness_value'), reverse=True)

            try:
                del population[self.num_individuals]
            except:
                pass

            # mutation params 12, 13
            for i in range(1, len(population)):
                if random.random() <= population[i].params[12]:
                    for idx in range(3):
                        if random.random() < population[i].params[13]:
                            if idx == 0:
                                population[i].params[12] = np.random.uniform(low=0, high=1, size=1)[0]
                            elif idx == 2:
                                population[i].params[13] = np.random.uniform(low=0, high=1, size=1)[0]
                            elif idx == 3:
                                population[i].params[13] = np.random.uniform(low=0, high=1, size=1)[0]

                if random.random() <= population[i].params[12]:
                    params = self.mutation(copy.deepcopy(population[i].params),
                                           elem_mutation_prob=copy.deepcopy(population[i].params[13]))
                    population[i] = IndividualDTO(id=str(uuid.uuid4()),
                                                  dataset=self.dataset,
                                                  params=[float(i) for i in params])  # TODO: check mutation
                evaluations_counter += 1

            population = estimate_fitness(population)

            ###
            logger.info("MUTATION IS OVER")

            population.sort(key=operator.attrgetter('fitness_value'), reverse=True)

            if evaluations_counter >= self.num_fitness_evaluations:
                bparams = ''.join([str(i) for i in population[0].params])
                logger.info(f"TERMINATION IS TRIGGERED."
                            f"THE BEST FITNESS {population[0].fitness_value}."
                            f"THE BEST PARAMS {bparams}.")
                return population[0].fitness_value

            current_fitness = population[0].fitness_value
            if (current_fitness > high_fitness) or (ii == 0):
                high_fitness = current_fitness

            bparams = ''.join([str(i) for i in population[0].params])
            logger.info(f"TERMINATION IS TRIGGERED."
                        f"THE BEST FITNESS {population[0].fitness_value}."
                        f"THE BEST PARAMS {bparams}.")

            x.append(ii)
            y.append(population[0].fitness_value)

            logger.info(f"Population len {len(population)}. "
                        f"Best params so far: {population[0].params}, with fitness: {population[0].fitness_value}.")

        logger.info(f"Y: {y}")

        return population[0].fitness_value


if __name__ == "__main__":
    run_algorithm()
