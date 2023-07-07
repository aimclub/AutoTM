#!/usr/bin/env python3
import copy
import logging
import os
import random
import sys
import uuid
from multiprocessing.pool import AsyncResult
from typing import List, Optional, Union

import click
import yaml
from hyperopt import STATUS_OK, fmin, hp, tpe
from tqdm import tqdm
from yaml import Loader

from autotm.algorithms_for_tuning.individuals import IndividualDTO
from autotm.utils import TqdmToLogger, make_log_config_dict

ALG_ID = "bo"

SPACE = {
    "decor": hp.quniform("decor", 0, 1e5, 0.05),
    "n1": hp.quniform("n1", 0, 8, 1),
    "spb": hp.quniform("spb", 1e-3, 1e2, 0.05),
    "stb": hp.quniform("stb", 1e-3, 1e2, 0.05),
    "n2": hp.quniform("n2", 0, 8, 1),
    "sp1": hp.quniform("sp1", -1e2, -1e-3, 0.05),
    "st1": hp.quniform("st1", -1e2, -1e-3, 0.05),
    "n3": hp.quniform("n3", 0, 8, 1),
    "sp2": hp.quniform("sp2", -1e2, -1e-3, 0.05),
    "st2": hp.quniform("st2", -1e2, -1e-3, 0.05),
    "n4": hp.quniform("n4", 0, 8, 1),
    "B": hp.quniform("B", 0, 8, 1),
    "decor_2": hp.quniform("decor_2", 0, 1e5, 0.05),
}

logger = logging.getLogger("BO")

# TODO: refactor this part
# getting config vars
if "FITNESS_CONFIG_PATH" in os.environ:
    filepath = os.environ["FITNESS_CONFIG_PATH"]
    with open(filepath, "r") as file:
        config = yaml.load(file, Loader=Loader)
else:
    config = {
        "testMode": False,
        "boAlgoParams": {
            "numEvals": 100
        }
    }


def estimate_fitness(
    population: List[IndividualDTO], _: bool = False, __: int = 2
) -> List[IndividualDTO]:
    results = []

    tqdm_out = TqdmToLogger(logger, level=logging.INFO)
    for p in tqdm(population, file=tqdm_out):
        individual = copy.deepcopy(p)
        individual.fitness_value = random.random()
        results.append(individual)

    return results


def log_best_solution(
    individual: IndividualDTO,
    wait_for_result_timeout: Optional[float] = None,
    alg_args: Optional[str] = None,
) -> Union[IndividualDTO, AsyncResult]:
    ind = copy.deepcopy(individual)
    ind.fitness_value = random.random()
    return ind


NUM_FITNESS_EVALUATIONS = config["boAlgoParams"]["numEvals"]


class BigartmFitness:
    def __init__(self, dataset: str, exp_id: Optional[int] = None):
        self.dataset = dataset
        self.exp_id = exp_id
        # self.best_solution: Optional[IndividualDTO] = None

    def parse_kwargs(self, **kwargs):
        params = []
        params.append(kwargs.get("decor", 1))
        params.append(int(kwargs.get("n1", 1)))
        params.append(kwargs.get("spb", 1))
        params.append(kwargs.get("stb", 1))
        params.append(int(kwargs.get("n2", 1)))
        params.append(kwargs.get("sp1", 1))
        params.append(kwargs.get("st1", 1))
        params.append(int(kwargs.get("n3", 1)))
        params.append(kwargs.get("sp2", 1))
        params.append(kwargs.get("st2", 1))
        params.append(int(kwargs.get("n4", 1)))
        params.append(int(kwargs.get("B", 1)))
        params.append(kwargs.get("decor_2", 1))
        return params

    def make_individ(self, **kwargs):
        # TODO: adapt this function to work with baesyian optimization
        params = [float(i) for i in self.parse_kwargs(**kwargs)]
        params = params[:-1] + [0.0, 0.0, 0.0] + [params[-1]]
        return IndividualDTO(
            id=str(uuid.uuid4()),
            dataset=self.dataset,
            params=params,
            exp_id=self.exp_id,
            alg_id=ALG_ID,
        )

    def __call__(self, kwargs):
        population = [self.make_individ(**kwargs)]

        population = estimate_fitness(population)
        individ = population[0]

        # if self.best_solution is None or individ.fitness_value > self.best_solution.fitness_value:
        #     self.best_solution = copy.deepcopy(individ)

        return {"loss": -1 * individ.fitness_value, "status": STATUS_OK}


@click.command(context_settings=dict(allow_extra_args=True))
@click.option("--dataset", required=True, type=str, help="dataset name in the config")
@click.option(
    "--log-file",
    type=str,
    default="/var/log/tm-alg-bo.log",
    help="a log file to write logs of the algorithm execution to",
)
@click.option("--exp-id", required=True, type=int, help="mlflow experiment id")
def run_algorithm(dataset, log_file, exp_id):
    run_uid = uuid.uuid4() if not config["testMode"] else None
    logging_config = make_log_config_dict(filename=log_file, uid=run_uid)
    logging.config.dictConfig(logging_config)

    fitness = BigartmFitness(dataset, exp_id)
    best_params = fmin(
        fitness, SPACE, algo=tpe.suggest, max_evals=NUM_FITNESS_EVALUATIONS
    )
    best_solution = fitness.make_individ(**best_params)
    best_solution = log_best_solution(
        best_solution, wait_for_result_timeout=-1, alg_args=" ".join(sys.argv)
    )
    print(best_solution.fitness_value * -1)


if __name__ == "__main__":
    run_algorithm()
