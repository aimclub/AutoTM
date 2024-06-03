import copy
import logging
import math
import time
import uuid
from abc import ABC, abstractmethod
from typing import List, Optional

import numpy as np

from autotm.abstract_params import AbstractParams
from autotm.algorithms_for_tuning.genetic_algorithm.statistics_collector import StatisticsCollector
from autotm.algorithms_for_tuning.genetic_algorithm.surrogate import Surrogate, set_surrogate_fitness, \
    get_prediction_uncertanty
from autotm.algorithms_for_tuning.individuals import Individual, IndividualBuilder
from autotm.fitness import local_tasks, cluster_tasks
from autotm.schemas import IndividualDTO

logger = logging.getLogger(__name__)


class FitnessEstimator:
    def __init__(self, num_fitness_evaluations: Optional[int] = None, statistics_collector: Optional[StatisticsCollector] = None):
        self._num_fitness_evaluations = num_fitness_evaluations
        self._evaluations_counter = 0
        self._statistics_collector = statistics_collector
        super().__init__()

    @property
    def num_fitness_evaluations(self) -> Optional[int]:
        return self._num_fitness_evaluations

    @property
    def evaluations_counter(self) -> int:
        return self._evaluations_counter

    @abstractmethod
    def fit(self, iter_num: int) -> None:
        ...

    @abstractmethod
    def log_best_solution(self,
                          individual: Individual,
                          wait_for_result_timeout: Optional[float] = None,
                          alg_args: Optional[str] = None,
                          is_tmp: bool = False) -> Individual:
        ...

    def estimate(self, iter_num: int, population: List[Individual]) -> List[Individual]:
        evaluated = [individual for individual in population if individual.dto.fitness_value is not None]
        not_evaluated = [individual for individual in population if individual.dto.fitness_value is None]
        evaluations_limit = max(0, self._num_fitness_evaluations - self._evaluations_counter) \
            if self._num_fitness_evaluations else len(not_evaluated)
        if len(not_evaluated) > evaluations_limit:
            not_evaluated = not_evaluated[:evaluations_limit]
        self._evaluations_counter += len(not_evaluated)
        new_evaluated = self._estimate(iter_num, not_evaluated)
        if self._statistics_collector:
            for individual in new_evaluated:
                self._statistics_collector.log_individual(individual)
        return evaluated + new_evaluated

    @abstractmethod
    def _estimate(self, iter_num: int, population: List[Individual]) -> List[Individual]:
        ...


class SurrogateEnabledFitnessEstimatorMixin(FitnessEstimator, ABC):
    SUPPORTED_CALC_SCHEMES = ["type1", "type2"]

    ibuilder: IndividualBuilder
    surrogate: Surrogate
    calc_scheme: str
    speedup: bool
    all_params: List[AbstractParams]
    all_fitness: List[float]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @staticmethod
    def surrogate_iteration(iter_num: int) -> bool:
        return (iter_num % 2 != 0) if iter_num > 0 else False

    def fit(self, iter_num: int) -> None:
        surrogate_iteration = self.surrogate_iteration(iter_num)
        if (self.calc_scheme == "type1" and not surrogate_iteration) or (self.calc_scheme == "type2"):
            self.surrogate.fit(np.array(self.all_params), np.array(self.all_fitness))

    def _estimate(self, iter_num: int, population: List[Individual]) -> List[Individual]:
        fitness_calc_time_start = time.time()
        surrogate_iteration = self.surrogate_iteration(iter_num)

        if not self.speedup or not surrogate_iteration or iter_num == -1:
            population = super().estimate(iter_num, population)
            self.save_params(population)

        if self.calc_scheme == "type1" and surrogate_iteration:
            population = self.surrogate_calculation(population)
        elif self.calc_scheme == "type2" and iter_num != -1:
            population = self._calculate_uncertain_res(iter_num, population)
            self.save_params(population)

        logger.info(f"TIME OF THE SURROGATE-BASED FITNESS FUNCTION: {time.time() - fitness_calc_time_start}")

        return population

    def surrogate_calculation(self, population: List[Individual]):
        x_val = np.array([copy.deepcopy(individ.params.to_vector()) for individ in population])
        y_pred = self.surrogate.predict(x_val)

        if not self.speedup:
            y_val = np.array([individ.fitness_value for individ in population])

            def check_val(fval):
                return not (fval is None or math.isnan(fval) or math.isinf(fval))

            def check_params(p):
                return all(check_val(el) for el in p)

            clean_params_and_f = []
            for i in range(len(y_val)):
                if not check_params(x_val[i]) or not check_val(y_val[i]):
                    logger.warning(
                        f"Bad params or fitness found. Fitness: {y_val[i]}. Params: {x_val[i]}."
                    )
                else:
                    clean_params_and_f.append((x_val[i], y_val[i]))

            x_val = clean_params_and_f[0]
            y_val = clean_params_and_f[1]
            r_2, mse, rmse = self.surrogate.score(x_val, y_val)
            logger.info(f"Real values: {list(y_val)}")
            logger.info(f"Predicted values: {list(y_pred)}")
            logger.info(f"R^2: {r_2}, MSE: {mse}, RMSE: {rmse}")

        for ix, individ in enumerate(population):
            individ.dto.fitness_value = set_surrogate_fitness(y_pred[ix])

        return population

    def _calculate_uncertain_res(self, iter_num: int, population: List[Individual], proc:float = 0.3):
        if len(population) == 0:
            return []

        x = np.array([individ.dto.params.to_vector() for individ in population])
        certanty = get_prediction_uncertanty(
            self.surrogate.surrogate, x, self.surrogate.name
        )
        recalculate_num = int(np.floor(len(certanty) * proc))
        logger.info(f"Certanty values: {certanty}")

        certanty, x = (
            list(t) for t in zip(*sorted(zip(certanty, x.tolist()), reverse=True))
        )  # check
        calculated = []
        for individual in population[:recalculate_num]:
            # copy
            individual_json = individual.dto.model_dump_json()
            individual = self.ibuilder.make_individual(dto=IndividualDTO.model_validate_json(individual_json))
            individual.dto.fitness_value = None
            calculated.append(individual)

        calculated = super().estimate(iter_num, calculated)

        self.all_params += [individ.dto.params.to_vector() for individ in calculated]
        self.all_fitness += [
            individ.dto.fitness_value["avg_coherence_score"] for individ in calculated
        ]

        pred_y = self.surrogate.predict(x[recalculate_num:])
        for ix, individual in enumerate(population[recalculate_num:]):
            dto = individual.dto
            dto = IndividualDTO(
                id=str(uuid.uuid4()),
                data_path=dto.data_path,
                params=dto.params,
                dataset=dto.dataset,
                fitness_value=set_surrogate_fitness(pred_y[ix]),
                exp_id=dto.exp_id,
                alg_id=dto.alg_id,
                topic_count=dto.topic_count,
                tag=dto.tag,
                train_option=dto.train_option,
            )
            calculated.append(self.ibuilder.make_individual(dto=dto))
        return calculated

    def save_params(self, population):
        params_and_f = [
            (copy.deepcopy(individ.params.to_vector()), individ.fitness_value)
            for individ in population
            if individ.fitness_value not in self.all_fitness
        ]

        def check_val(fval):
            return not (fval is None or math.isnan(fval) or math.isinf(fval))

        def check_params(pp):
            return all(check_val(el) for el in pp)

        clean_params_and_f = []
        for p, f in params_and_f:
            if not check_params(p) or not check_val(f):
                logger.warning(f"Bad params or fitness found. Fitness: {f}. Params: {p}.")
            else:
                clean_params_and_f.append((p, f))

        pops = [p for p, _ in clean_params_and_f]
        fs = [f for _, f in clean_params_and_f]

        self.all_params += pops
        self.all_fitness += fs


class ComputableFitnessEstimator(FitnessEstimator):
    def __init__(self,
                 ibuilder: IndividualBuilder,
                 num_fitness_evaluations: Optional[int] = None,
                 statistics_collector: Optional[StatisticsCollector] = None):
        self.ibuilder = ibuilder
        super().__init__(num_fitness_evaluations, statistics_collector)

    def fit(self, iter_num: int) -> None:
        pass

    def log_best_solution(self,
                          individual: Individual,
                          wait_for_result_timeout: Optional[float] = None,
                          alg_args: Optional[str] = None,
                          is_tmp: bool = False) -> Individual:
        return local_tasks.log_best_solution(self.ibuilder, individual, wait_for_result_timeout, alg_args, is_tmp)

    def _estimate(self, iter_num: int, population: List[Individual]) -> List[Individual]:
        return local_tasks.estimate_fitness(self.ibuilder, population)


class DistributedComputableFitnessEstimator(FitnessEstimator):
    def __init__(self,
                 ibuilder: IndividualBuilder,
                 num_fitness_evaluations: Optional[int] = None,
                 statistics_collector: Optional[StatisticsCollector] = None):
        self.app = cluster_tasks.make_celery_app()
        self.ibuilder = ibuilder
        super().__init__(num_fitness_evaluations, statistics_collector)

    def fit(self, iter_num: int) -> None:
        pass

    def log_best_solution(self,
                          individual: Individual,
                          wait_for_result_timeout: Optional[float] = None,
                          alg_args: Optional[str] = None,
                          is_tmp: bool = False) -> Individual:
        return cluster_tasks.log_best_solution(self.ibuilder, individual,
                                               wait_for_result_timeout, alg_args, is_tmp, app=self.app)

    def _estimate(self, iter_num: int, population: List[Individual]) -> List[Individual]:
        return cluster_tasks.parallel_fitness(self.ibuilder, population, app=self.app)


class SurrogateEnabledComputableFitnessEstimator(ComputableFitnessEstimator, SurrogateEnabledFitnessEstimatorMixin):
    def __init__(self,
                 ibuilder: IndividualBuilder,
                 surrogate: Surrogate,
                 calc_scheme: str,
                 speedup: bool = True,
                 num_fitness_evaluations: Optional[int] = None,
                 statistics_collector: Optional[StatisticsCollector] = None):
        self.ibuilder = ibuilder
        self.surrogate = surrogate
        self.calc_scheme = calc_scheme
        self.speedup = speedup

        self.all_params: List[AbstractParams] = []
        self.all_fitness: List[float] = []

        if calc_scheme not in self.SUPPORTED_CALC_SCHEMES:
            raise ValueError(f"Unexpected surrogate scheme! {self.calc_scheme}")
        super().__init__(ibuilder, num_fitness_evaluations, statistics_collector)


class DistributedSurrogateEnabledComputableFitnessEstimator(
    DistributedComputableFitnessEstimator,
    SurrogateEnabledFitnessEstimatorMixin
):
    def __init__(self,
                 ibuilder: IndividualBuilder,
                 surrogate: Surrogate,
                 calc_scheme: str,
                 speedup: bool = True,
                 num_fitness_evaluations: Optional[int] = None,
                 statistics_collector: Optional[StatisticsCollector] = None):
        self.ibuilder = ibuilder
        self.surrogate = surrogate
        self.calc_scheme = calc_scheme
        self.speedup = speedup

        self.all_params: List[AbstractParams] = []
        self.all_fitness: List[float] = []

        if calc_scheme not in self.SUPPORTED_CALC_SCHEMES:
            raise ValueError(f"Unexpected surrogate scheme! {self.calc_scheme}")
        super().__init__(ibuilder, num_fitness_evaluations, statistics_collector)
