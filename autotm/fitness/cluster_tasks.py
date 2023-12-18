import logging
import os
import time
import uuid
from multiprocessing.process import current_process
from typing import List, Optional, Union, cast

import celery
from billiard.exceptions import SoftTimeLimitExceeded
from celery import shared_task, group, Task
from celery.exceptions import TimeLimitExceeded
from celery.result import GroupResult, AsyncResult
from celery.utils.log import get_task_logger
from tqdm import tqdm

from autotm.algorithms_for_tuning.individuals import Individual, make_individual
from autotm.fitness.tm import fit_tm_of_individual
from autotm.params_logging_utils import model_files, log_params_and_artifacts, log_stats
from autotm.schemas import IndividualDTO
from autotm.utils import TqdmToLogger, AVG_COHERENCE_SCORE

logger = logging.getLogger("root")
task_logger = get_task_logger(__name__)


class RemotelyCalculableIndividual:
    def get_params(self) -> str:
        raise NotImplementedError()

    def set_fitness(self, value):
        raise NotImplementedError()


def make_celery_app(client_to_broker_max_retries=3):
    app = celery.Celery(
        "distributed_fitness",
        backend=os.environ.get("CELERY_RESULT_BACKEND"),
        broker=os.environ["CELERY_BROKER_URL"]
    )

    # for explanation of this options see:
    # 1. https://stackoverflow.com/questions/56805193/how-to-set-up-celery-producer-send-task-timeout
    # 2. https://github.com/celery/celery/issues/4627#issuecomment-396907957
    app.conf.update(
        task_acks_late=True,
        task_reject_on_worker_lost=True,
        task_acks_on_failure_or_timeout=False,
        worker_prefetch_multiplier=1,
        broker_transport_options={
            "max_retries": client_to_broker_max_retries,
            "interval_start": 0,
            "interval_step": 0.2,
            "interval_max": 0.5
        }
    )

    return app


def do_fitness_calculating(individual: str,
                           log_artifact_and_parameters: bool = False,
                           log_run_stats: bool = False,
                           alg_args: Optional[str] = None) -> str:
    individual: IndividualDTO = IndividualDTO.parse_raw(individual)
    task_logger.info(f"Calculating fitness for an individual with id {individual.id}: \n{individual}")

    with fit_tm_of_individual(
            dataset=individual.dataset,
            data_path=individual.data_path,
            params=individual.params,
            fitness_name=individual.fitness_name,
            topic_count=individual.topic_count,
            force_dataset_settings_checkout=individual.force_dataset_settings_checkout,
            train_option=individual.train_option) as (time_metrics, metrics, tm):
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
        self.retry(max_retries=1, countdown=5)


def parallel_fitness(population: List[Individual],
                     use_tqdm: bool = False,
                     tqdm_check_period: int = 2,
                     app: Optional[celery.Celery] = None) -> List[Individual]:
    ids = [ind.dto.id for ind in population]
    assert len(set(ids)) == len(population), \
        f"There are individuals with duplicate ids: {ids}"

    logger.info("Calculating fitness...")
    logger.info(f"Sending individuals to be calculated with uids: {[p.dto.id for p in population]}")

    fitness_tasks = []
    for individual in population:
        task = cast(
            Task,
            calculate_fitness.signature((individual.dto.json(), False, False), options={"queue": "fitness_tasks"})
        )
        # todo: add it later
        # if app is not None:
        #     task.bind(app)
        fitness_tasks.append(task)

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
    return [make_individual(results_by_id[ind.dto.id]) for ind in population]


def log_best_solution(individual: Individual,
                      wait_for_result_timeout: Optional[float] = None,
                      alg_args: Optional[str] = None,
                      is_tmp: bool = False,
                      app: Optional[celery.Celery] = None) \
        -> Individual:
    if is_tmp:
        return make_individual(individual.dto)
    # ind = fitness_to_json(individual)
    ind = individual.dto.json()
    logger.info(f"Sending a best individual to be logged: {ind}")
    task: Task = calculate_fitness.signature(
        (ind, True, False, alg_args),
        options={"queue": "fitness_tasks"}
    )

    # todo: add it later
    # if app is not None:
    #     task.bind(app)

    result: AsyncResult = task.apply_async()
    logger.debug(f"Started a task for best individual logging with id: {result.task_id}")

    # if wait_for_result_timeout:
    #     # it may block forever
    #     timeout = wait_for_result_timeout if wait_for_result_timeout > 0 else None
    #     r = result.get(timeout=timeout)
    #     # r = fitness_from_json(r)
    #     r = IndividualDTO.parse_raw(r)
    # else:
    #     r = result

    r = result.get(timeout=wait_for_result_timeout)
    r = IndividualDTO.parse_raw(r)
    ind = make_individual(r)

    return ind


class FitnessCalculatorWrapper:
    def __init__(self, dataset, data_path, topic_count, train_option):
        self.dataset = dataset
        self.data_path = data_path
        self.topic_count = topic_count
        self.train_option = train_option

    def run(self, params):
        params = list(params)
        params = params[:-1] + [0, 0, 0] + [params[-1]]

        solution_dto = IndividualDTO(id=str(uuid.uuid4()),
                                     dataset=self.dataset,
                                     params=params,
                                     alg_id="ga",
                                     topic_count=self.topic_count, train_option=self.train_option)

        dto = parallel_fitness([solution_dto])[0]
        result = dto.fitness_value[AVG_COHERENCE_SCORE]
        return -result
