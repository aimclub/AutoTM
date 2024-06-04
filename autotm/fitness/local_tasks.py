import logging
from typing import List, Optional

from autotm.algorithms_for_tuning.individuals import Individual, IndividualBuilder
from autotm.fitness.tm import fit_tm_of_individual
from autotm.params_logging_utils import log_params_and_artifacts, log_stats, model_files
from autotm.schemas import IndividualDTO

logger = logging.getLogger("root")


def do_fitness_calculating(
        individual: IndividualDTO,
        log_artifact_and_parameters: bool = False,
        log_run_stats: bool = False,
        alg_args: Optional[str] = None,
        is_tmp: bool = False,
) -> IndividualDTO:
    # make copy
    individual_json = individual.model_dump_json()
    individual = IndividualDTO.model_validate_json(individual_json)
    logger.info("Doing fitness calculating with individual %s" % individual_json)

    with fit_tm_of_individual(
            dataset=individual.dataset,
            data_path=individual.data_path,
            params=individual.params,
            fitness_name=individual.fitness_name,
            topic_count=individual.topic_count,
            force_dataset_settings_checkout=individual.force_dataset_settings_checkout,
            train_option=individual.train_option,
    ) as (time_metrics, metrics, tm):
        individual.fitness_value = metrics

        with model_files(tm) as tm_files:
            if log_artifact_and_parameters:
                log_params_and_artifacts(
                    tm, tm_files, individual, time_metrics, alg_args, is_tmp=is_tmp
                )

            if log_run_stats:
                log_stats(tm, tm_files, individual, time_metrics, alg_args)

    return individual


def calculate_fitness(
        individual: IndividualDTO,
        log_artifact_and_parameters: bool = False,
        log_run_stats: bool = False,
        alg_args: Optional[str] = None,
        is_tmp: bool = False,
) -> IndividualDTO:
    return do_fitness_calculating(
        individual, log_artifact_and_parameters, log_run_stats, alg_args, is_tmp
    )


def estimate_fitness(ibuilder: IndividualBuilder, population: List[Individual]) -> List[Individual]:
    logger.info("Calculating fitness...")
    population_with_fitness = []
    for individual in population:
        if individual.dto.fitness_value is not None:
            logger.info("Fitness value already calculated")
            population_with_fitness.append(individual)
            continue
        individ_with_fitness = calculate_fitness(individual.dto)
        population_with_fitness.append(ibuilder.make_individual(individ_with_fitness))
    logger.info("The fitness results have been obtained")
    return population_with_fitness


def log_best_solution(
        ibuilder: IndividualBuilder,
        individual: Individual,
        wait_for_result_timeout: Optional[float] = None,
        alg_args: Optional[str] = None,
        is_tmp: bool = False,
):
    logger.info("Sending a best individual to be logged")
    res = ibuilder.make_individual(calculate_fitness(individual.dto,
                                            log_artifact_and_parameters=True,
                                            is_tmp=is_tmp))

    # TODO: write logging
    return res
