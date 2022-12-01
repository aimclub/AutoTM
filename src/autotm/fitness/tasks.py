import logging
import time
import uuid
from multiprocessing.process import current_process
from typing import List, Optional, Union
import json

from tqdm import tqdm

from autotm.params_logging_utils import log_params_and_artifacts, log_stats, model_files
from autotm.fitness.tm import fit_tm_of_individual
from autotm.utils import TqdmToLogger
from autotm.schemas import IndividualDTO, fitness_to_json, fitness_from_json

logger = logging.getLogger("root")


# task_logger = get_task_logger(__name__)

def do_fitness_calculating(individual: str,
                           log_artifact_and_parameters: bool = False,
                           log_run_stats: bool = False,
                           alg_args: Optional[str] = None) -> str:
    individual: IndividualDTO = IndividualDTO.parse_raw(individual)
    # task_logger.info(f"Calculating fitness for an individual with id {individual.id}: \n{individual}")

    with fit_tm_of_individual(
            dataset=individual.dataset,
            data_path=individual.data_path,
            params=individual.params,
            fitness_name=individual.fitness_name,
            topic_count=individual.topic_count,
            force_dataset_settings_checkout=individual.force_dataset_settings_checkout) as (time_metrics, metrics, tm):
        individual.fitness_value = metrics

        with model_files(tm) as tm_files:
            if log_artifact_and_parameters:
                log_params_and_artifacts(tm, tm_files, individual, time_metrics, alg_args)

            if log_run_stats:
                log_stats(tm, tm_files, individual, time_metrics, alg_args)

    # task_logger.info(f"Fitness has been calculated for {individual.id}: {individual.fitness_value}")

    return individual.json()


def calculate_fitness(individual: str,
                      log_artifact_and_parameters: bool = False,
                      log_run_stats: bool = False,
                      alg_args: Optional[str] = None) -> str:
    try:
        return do_fitness_calculating(individual, log_artifact_and_parameters, log_run_stats, alg_args)
    except:
        raise Exception(f"Some exception")


def estimate_fitness(population: List[IndividualDTO]) -> List[IndividualDTO]:
    # ids = [ind.id for ind in population]
    # assert len(set(ids)) == len(population), \
    #     f"There are individuals with duplicate ids: {ids}"
    logger.info("Calculating fitness...")
    population_with_fitness = []
    for individual in population:
        individual.dto = fitness_from_json(calculate_fitness(fitness_to_json(individual.dto)))
        population_with_fitness.append(individual)
    logger.info("The fitness results have been obtained")
    return population_with_fitness


def log_best_solution(individual: IndividualDTO,
                      wait_for_result_timeout: Optional[float] = None,
                      alg_args: Optional[str] = None):
    ind = individual.json()
    logger.info(f"Sending a best individual to be logged: {ind}")
    logger.debug(f"Started a task for best individual logging with id: {individual.id}")
    # TODO: write logging
    return ind
