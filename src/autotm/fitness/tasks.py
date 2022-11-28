import logging
import os
import time
import uuid
from multiprocessing.process import current_process
from typing import List, Optional, Union

from billiard.exceptions import SoftTimeLimitExceeded
from tqdm import tqdm

from kube_fitness.params_logging_utils import log_params_and_artifacts, log_stats, model_files
from kube_fitness.schemas import IndividualDTO, fitness_to_json, fitness_from_json
from kube_fitness.tm import fit_tm_of_individual
from kube_fitness.utils import TqdmToLogger


def do_fitness_calculating(individual: str,
                           log_artifact_and_parameters: bool = False,
                           log_run_stats: bool = False,
                           alg_args: Optional[str] = None) -> str:
    individual: IndividualDTO = IndividualDTO.parse_raw(individual)
    task_logger.info(f"Calculating fitness for an individual with id {individual.id}: \n{individual}")

    with fit_tm_of_individual(
            dataset=individual.dataset,
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

    task_logger.info(f"Fitness has been calculated for {individual.id}: {individual.fitness_value}")

    return individual.json()


# @shared_task(time_limit=25*60, soft_time_limit=20*60, autoretry_for=(Exception,),
# retry_kwargs={'max_retries': 3, 'countdown': 5})
@shared_task(bind=True, time_limit=25*60, soft_time_limit=20*60)
def calculate_fitness(self: Task,
                      individual: str,
                      log_artifact_and_parameters: bool = False,
                      log_run_stats: bool = False,
                      alg_args: Optional[str] = None) -> str:
    # A hack to solve the problem 'AssertionError: daemonic processes are not allowed to have children'
    # see the discussion: https://github.com/celery/celery/issues/1709
    # noinspection PyUnresolvedReferences
    current_process()._config['daemon'] = False

    try:
        return do_fitness_calculating(individual, log_artifact_and_parameters, log_run_stats, alg_args)
    except TimeLimitExceeded as ex:
        logger.error("Encountered hard time limit!")
    except SoftTimeLimitExceeded as ex:
        raise Exception(f"Soft time limit encountered") from ex
    except Exception:
        self.retry(max_retries=3, countdown=5)


def parallel_fitness(population: List[IndividualDTO],
                     use_tqdm: bool = False,
                     tqdm_check_period: int = 2) -> List[IndividualDTO]:
    ids = [ind.id for ind in population]
    assert len(set(ids)) == len(population), \
        f"There are individuals with duplicate ids: {ids}"

    logger.info("Calculating fitness...")
    logger.info(f"Sending individuals to be calculated with uids: {[p.id for p in population]}")

    fitness_tasks = [
        calculate_fitness.signature(
            # (fitness_to_json(individual), False, True),
            (individual.json(), False, True),
            options={"queue": "fitness_tasks"}
        ) for individual in population
    ]

    g: GroupResult = group(*fitness_tasks).apply_async()

    logger.info(f"Corresponding celery tasks ids: {[child.id for child in g.children]}")

    tqdm_out = TqdmToLogger(logger, level=logging.INFO)

    if use_tqdm:
        # TODO: add timeout here
        with tqdm(total=len(fitness_tasks), file=tqdm_out) as pbar:
            while True:
                cc = g.completed_count()
                # logger.debug(f"Completed task count : {cc}")
                pbar.update(cc - pbar.n)

                # logger.debug(f"is ready {g.ready()} or failed {g.failed()}")

                if g.ready() or g.failed():
                    break

                # TODO: make it into parameter
                time.sleep(tqdm_check_period)

    # TODO: ugly workround to solve indefinite hanging g.get(), see pages below for additional info
    # https://stackoverflow.com/questions/63860955/celery-async-result-get-hangs-waiting-for-result-even-after-celery-worker-has
    # https://stackoverflow.com/questions/49006182/celery-redis-get-hangs-indefinitely-after-running-smoothly-for-70-hours
    logger.info("Getting results")
    # TODO: this is commented because of https://github.com/celery/celery/pull/7040
    # waiting_delay = int(os.environ.get('AUTOTM_KUBE_FITNESS_WAITING_DELAY', '90'))
    # while not g.ready():
    #     logger.info(f"Results are not ready. Waiting for the next {waiting_delay} seconds...")
    #     time.sleep(waiting_delay)
    # logger.info("Results are ready, trying ot obtain them...")
    # results = g.get()
    results = g.get()
    logger.info("The results have been obtained")

    # restoring the order in the resulting population according to the initial population
    # results_by_id = {ind.id: ind for ind in (fitness_from_json(r) for r in results)}
    results_by_id = {ind.id: ind for ind in (IndividualDTO.parse_raw(r) for r in results)}
    return [results_by_id[ind.id] for ind in population]


def log_best_solution(individual: IndividualDTO,
                      wait_for_result_timeout: Optional[float] = None,
                      alg_args: Optional[str] = None) \
        -> Union[IndividualDTO, AsyncResult]:
    # ind = fitness_to_json(individual)
    ind = individual.json()
    logger.info(f"Sending a best individual to be logged: {ind}")
    task: Task = calculate_fitness.signature(
        (ind, True, True, alg_args),
        options={"queue": "fitness_tasks"}
    )

    result: AsyncResult = task.apply_async()
    logger.debug(f"Started a task for best individual logging with id: {result.task_id}")

    if wait_for_result_timeout:
        # it may block forever
        timeout = wait_for_result_timeout if wait_for_result_timeout > 0 else None
        r = result.get(timeout=timeout)
        # r = fitness_from_json(r)
        r = IndividualDTO.parse_raw(r)
    else:
        r = result

    return r

