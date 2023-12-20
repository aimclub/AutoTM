import logging
from typing import List, Optional

from autotm.params_logging_utils import log_params_and_artifacts, log_stats, model_files
from autotm.fitness.tm import fit_tm_of_individual
from autotm.schemas import IndividualDTO, fitness_to_json, fitness_from_json
from autotm.algorithms_for_tuning.individuals import make_individual, Individual

logger = logging.getLogger("root")


def do_fitness_calculating(
    individual: str,
    log_artifact_and_parameters: bool = False,
    log_run_stats: bool = False,
    alg_args: Optional[str] = None,
    is_tmp: bool = False,
) -> str:
    logger.info("Doing fitness calculating with individual %s" % individual)
    individual: IndividualDTO = IndividualDTO.parse_raw(individual)

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

    return individual.json()


def calculate_fitness(
    individual: str,
    log_artifact_and_parameters: bool = False,
    log_run_stats: bool = False,
    alg_args: Optional[str] = None,
    is_tmp: bool = False,
) -> str:
    try:
        return do_fitness_calculating(
            individual, log_artifact_and_parameters, log_run_stats, alg_args, is_tmp
        )
    except Exception as e:
        print(str(e))
        raise Exception("Some exception")


def estimate_fitness(population: List[Individual]) -> List[Individual]:
    logger.info("Calculating fitness...")
    population_with_fitness = []
    for individual in population:
        json_individ = fitness_to_json(individual.dto)
        individ_with_fitness = calculate_fitness(json_individ)
        population_with_fitness.append(
            make_individual(fitness_from_json(individ_with_fitness))
        )
    logger.info("The fitness results have been obtained")
    return population_with_fitness


def log_best_solution(
    individual: Individual,
    wait_for_result_timeout: Optional[float] = None,
    alg_args: Optional[str] = None,
    is_tmp: bool = False,
) -> Individual:
    logger.info("Sending a best individual to be logged")
    res = make_individual(
        fitness_from_json(
            calculate_fitness(
                fitness_to_json(individual.dto),
                log_artifact_and_parameters=True,
                is_tmp=is_tmp,
            )
        )
    )

    # TODO: write logging
    return res
