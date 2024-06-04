#!/usr/bin/env python3
import copy
import logging
import logging.config
import os
import random
import sys
import uuid
from multiprocessing.pool import AsyncResult
from typing import List, Optional, Union

import yaml
from hyperopt import STATUS_OK, fmin, hp, tpe
from tqdm import tqdm
from yaml import Loader

from autotm.algorithms_for_tuning.individuals import IndividualDTO, IndividualBuilder
from autotm.fitness.estimator import FitnessEstimator, ComputableFitnessEstimator
from autotm.fitness.tm import fit_tm, TopicModel
from autotm.params import FixedListParams
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
    def __init__(self,
                 data_path: str,
                 topic_count: int,
                 ibuilder: IndividualBuilder,
                 fitness_estimator: FitnessEstimator,
                 dataset: str,
                 exp_id: Optional[int] = None):
        self.data_path = data_path
        self.topic_count = topic_count
        self.ibuilder = ibuilder
        self.fitness_estimator = fitness_estimator
        self.dataset = dataset
        self.exp_id = exp_id

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

    def make_ind_dto(self, **kwargs):
        # TODO: adapt this function to work with baesyian optimization
        params = [float(i) for i in self.parse_kwargs(**kwargs)]
        params = params[:-1] + [0.0, 0.0, 0.0] + [params[-1]]
        return IndividualDTO(
            id=str(uuid.uuid4()),
            data_path=self.data_path,
            dataset=self.dataset,
            topic_count=self.topic_count,
            params=FixedListParams(params=params),
            exp_id=self.exp_id,
            alg_id=ALG_ID,
        )

    def __call__(self, kwargs):
        population = [self.ibuilder.make_individual(self.make_ind_dto(**kwargs))]
        population = self.fitness_estimator.estimate(-1, population)
        individ = population[0]
        return {"loss": -1 * individ.fitness_value, "status": STATUS_OK}


def run_algorithm(dataset,
                  data_path,
                  topic_count,
                  log_file,
                  exp_id,
                  num_evaluations,
                  individual_type: str = "regular",
                  train_option: str = "offline") -> TopicModel:
    run_uid = uuid.uuid4() if not config["testMode"] else None
    logging_config = make_log_config_dict(filename=log_file, uid=run_uid)
    logging.config.dictConfig(logging_config)

    ibuilder = IndividualBuilder(individual_type)
    fitness_estimator = ComputableFitnessEstimator(ibuilder, num_evaluations)

    fitness = BigartmFitness(data_path, topic_count, ibuilder, fitness_estimator, dataset, exp_id)
    best_params = fmin(
        fitness, SPACE, algo=tpe.suggest, max_evals=num_evaluations
    )
    best_solution_dto = fitness.make_ind_dto(**best_params)
    best_solution_dto = log_best_solution(
        best_solution_dto, wait_for_result_timeout=-1, alg_args=" ".join(sys.argv)
    )

    best_topic_model = fit_tm(
        preproc_data_path=data_path,
        topic_count=topic_count,
        params=best_solution_dto.params,
        train_option=train_option
    )

    return best_topic_model
