import copy
import logging
import os
import random
from typing import List, Optional

import yaml
from kube_fitness.utils import TqdmToLogger

from algorithms_for_tuning.individuals import Individual, make_individual

logger = logging.getLogger("GA_algo")


def test_mode_from_env() -> bool:
    if "AUTOTM_TEST_MODE" not in os.environ:
        return False
    test_mode = os.environ["AUTOTM_TEST_MODE"]

    return test_mode.lower() in ("yes", "true", "t", "1")

def test_mode_from_config() -> bool:
    if "FITNESS_CONFIG_PATH" in os.environ:
        filepath = os.environ["FITNESS_CONFIG_PATH"]
    else:
        filepath = "../../algorithms_for_tuning/genetic_algorithm/config.yaml"

    if os.path.exists(filepath):
        with open(filepath, "r") as file:
            config = yaml.load(file, Loader=yaml.Loader)
    else:
        config = dict()

    return config.get('testMode', False)


test_mode = test_mode_from_env() or test_mode_from_config()


if not test_mode:
    from kube_fitness.tasks import make_celery_app as prepare_fitness_estimator
    from kube_fitness.tasks import parallel_fitness
    from kube_fitness.tasks import log_best_solution as lbs

    def estimate_fitness(population: List[Individual],
                         use_tqdm: bool = False,
                         tqdm_check_period: int = 2) -> List[Individual]:

        population_dtos = [p.dto for p in population]
        results_dto = parallel_fitness(population_dtos, use_tqdm, tqdm_check_period)
        results = [make_individual(dto=dto) for dto in results_dto]

        return results

    def log_best_solution(individual: Individual, alg_args: Optional[str] = None):
        lbs(individual.dto, alg_args=alg_args)
else:
    from tqdm import tqdm

    def prepare_fitness_estimator():
        pass

    def estimate_fitness(population: List[Individual],
                         use_tqdm: bool = False,
                         tqdm_check_period: int = 2) -> List[Individual]:
        results = []

        tqdm_out = TqdmToLogger(logger, level=logging.INFO)
        for p in tqdm(population, file=tqdm_out):
            individual = copy.deepcopy(p)
            individual.fitness_value = random.random()
            results.append(individual)

        return results


    def log_best_solution(individual: Individual, alg_args: Optional[str] = None):
        pass
