import copy
import gc
import logging
import operator
import random
import sys
import time
import uuid
from typing import Optional, Callable, List

import numpy as np

from autotm.algorithms_for_tuning.genetic_algorithm.selection import selection
from autotm.algorithms_for_tuning.genetic_algorithm.statistics_collector import StatisticsCollector
from autotm.algorithms_for_tuning.individuals import IndividualDTO, Individual, IndividualBuilder
from autotm.algorithms_for_tuning.nelder_mead_optimization.nelder_mead import (
    NelderMeadOptimization,
)
from autotm.fitness.estimator import FitnessEstimator
from autotm.params import create_individual
from autotm.utils import AVG_COHERENCE_SCORE
from autotm.visualization.dynamic_tracker import MetricsCollector

ALG_ID = "ga"

logger = logging.getLogger("GA_algo")


def run_with_retry(action: Callable[[], object],
                   condition: Callable[[object], bool],
                   default_value: object = None,
                   max_retries: int = 5):
    for _ in range(max_retries):
        value = action()
        if condition(value):
            return value
    logger.warning(f"Cannot perform action after {max_retries} retries")
    return default_value


class ModelStorage:
    def __init__(self):
        self.stage_1_components = {}  # {config_id: id}
        self.stage_1_hyperp = {}  # {config_id: [[params1, params2]]}

    def model_search(self, model):
        raise NotImplementedError
        # for model.components


class GA:
    def __init__(
            self,
            dataset,
            data_path,
            num_individuals,
            num_iterations,
            ibuilder: IndividualBuilder,
            fitness_estimator: FitnessEstimator,
            mutation_type="mutation_one_param",
            crossover_type="blend_crossover",
            selection_type="fitness_prop",
            elem_cross_prob=0.2,
            early_stopping_iterations: Optional[int] = 500,
            best_proc=0.3,
            alpha=None,
            exp_id: Optional[int] = None,
            topic_count: Optional[int] = None,
            fitness_obj_type="single_objective",
            tag: Optional[str] = None,
            use_pipeline: bool = False,
            use_nelder_mead: bool = False,
            use_nelder_mead_in_mutation: bool = False,
            use_nelder_mead_in_crossover: bool = False,
            use_nelder_mead_in_selector: bool = False,
            train_option: str = "offline",
            statistics_collector: Optional[StatisticsCollector] = None,
            **kwargs,
    ):
        """

        :param dataset: dataset name
        :param data_path: path to data
        :param num_individuals: number of individuals
        :param num_iterations: number of iterations
        :param mutation_type: type of mutation, available types ['mutation_one_param', 'combined', 'psm', 'positioning_mutation']
        :param crossover_type: type of crossover, available types ['crossover_pmx', 'crossover_one_point', 'blend_crossover']
        :param selection_type: type of selection, available types ['fitness_prop', 'rank_based']
        :param elem_cross_prob: probability of crossover
        :param num_fitness_evaluations: number of fitness evaluations in case of limited resources
        :param early_stopping_iterations: number of iterations when there is no significant changes in fitness to stop training
        :param best_proc: percentage of parents to be transferred to new generation
        :param alpha:
        :param exp_id:
        :param surrogate_name:
        :param calc_scheme: how to apply surrogates, available values: 'type 1' - to use surrogates through iteration, 'type 2' - calculating 70% on each iteration
        :param topic_count:
        :param fitness_obj_type:
        :param tag:
        :param use_nelder_mead: should Nelder-Mead enchan
        :param kwargs:
        """

        self.dataset = dataset

        if crossover_type == "blend_crossover":
            self.crossover_children = 1
        else:
            self.crossover_children = 2
        self.data_path = data_path
        self.num_individuals = num_individuals
        self.num_iterations = num_iterations
        self.ibuilder = ibuilder
        self.fitness_estimator = fitness_estimator
        self.mutation_type = mutation_type
        self.crossover_type = crossover_type
        self.selection = selection(selection_type)
        self.elem_cross_prob = elem_cross_prob
        self.alpha = alpha
        self.early_stopping_iterations = early_stopping_iterations
        self.fitness_obj_type = fitness_obj_type
        self.best_proc = best_proc
        self.exp_id = exp_id
        self.topic_count = topic_count
        self.tag = tag
        self.use_pipeline = use_pipeline
        self.statistics_collector = statistics_collector
        # hyperparams
        self.use_nelder_mead = use_nelder_mead
        self.use_nelder_mead_in_mutation = use_nelder_mead_in_mutation
        self.use_nelder_mead_in_crossover = use_nelder_mead_in_crossover
        self.use_nelder_mead_in_selectior = use_nelder_mead_in_selector
        self.train_option = train_option
        self.metric_collector = MetricsCollector(
            dataset=self.dataset, n_specific_topics=topic_count
        )
        self.crossover_changes_dict = (
            {}
        )  # generation, parent_1_params, parent_2_params, ...

    def init_population(self):
        list_of_individuals = []
        for i in range(self.num_individuals):
            dto = IndividualDTO(
                id=str(uuid.uuid4()),
                data_path=self.data_path,
                dataset=self.dataset,
                params=create_individual(base_model=i == 0, use_pipeline=self.use_pipeline),
                exp_id=self.exp_id,
                alg_id=ALG_ID,
                iteration_id=0,
                topic_count=self.topic_count,
                tag=self.tag,
                train_option=self.train_option,
            )
            # TODO: improve heuristic on search space
            list_of_individuals.append(self.ibuilder.make_individual(dto=dto))

        population_with_fitness = self.run_fitness(list_of_individuals, -1)

        # self.save_params(population_with_fitness)
        # if self.surrogate is not None and self.calc_scheme == "type2":
        #     self.surrogate.fit(np.array(self.all_params), np.array(self.all_fitness))
        #     logger.info("Surrogate is initialized!")

        self.fitness_estimator.fit(iter_num=-1)

        return population_with_fitness

    @staticmethod
    def _sort_population(population):
        population.sort(key=operator.attrgetter("fitness_value"), reverse=True)

    def run_crossover(self, pairs_generator, iteration_num: int):
        new_generation = []

        crossover_changes = {
            "parent_1_params": [],
            "parent_2_params": [],
            "parent_1_fitness": [],
            "parent_2_fitness": [],
            "child_id": [],
        }

        for i, j in pairs_generator:
            if i is None:
                break

            children = run_with_retry(
                action=lambda: i.params.crossover(j.params, crossover_type=self.crossover_type,
                                                  elem_cross_prob=self.elem_cross_prob, alpha=self.alpha),
                condition=lambda candidates: all(child.validate_params() for child in candidates),
                default_value=[]
            )

            children_dto = [IndividualDTO(
                id=str(uuid.uuid4()),
                data_path=self.data_path,
                dataset=self.dataset,
                params=child,
                exp_id=self.exp_id,
                alg_id=ALG_ID,
                iteration_id=iteration_num,
                topic_count=self.topic_count,
                tag=self.tag,
                train_option=self.train_option,
            ) for child in children]

            individuals = [self.ibuilder.make_individual(child) for child in children_dto]
            new_generation += individuals

            crossover_changes["parent_1_params"].append(i.params)
            crossover_changes["parent_2_params"].append(j.params)
            crossover_changes["parent_1_fitness"].append(i.fitness_value)
            crossover_changes["parent_2_fitness"].append(j.fitness_value)
            crossover_changes["child_id"].append(len(new_generation) - 1)

        if len(new_generation) > 0:
            new_generation = self.run_fitness(new_generation, iteration_num)
            logger.info(f"size of the new generation is {len(new_generation)}")

            for i in range(len(crossover_changes["parent_1_params"])):
                child_index = crossover_changes["child_id"][i]
                if child_index >= len(new_generation):
                    continue
                self.metric_collector.save_crossover(
                    generation=iteration_num,
                    parent_1=crossover_changes["parent_1_params"][i],
                    parent_2=crossover_changes["parent_2_params"][i],
                    parent_1_fitness=crossover_changes["parent_1_fitness"][i],
                    parent_2_fitness=crossover_changes["parent_2_fitness"][i],
                    child_1=new_generation[child_index].params,
                    child_1_fitness=new_generation[child_index].fitness_value,
                )

        return new_generation

    def apply_nelder_mead(self, starting_points_set, num_gen, num_iterations=2):
        nelder_opt = NelderMeadOptimization(
            data_path=self.data_path,
            dataset=self.dataset,
            exp_id=self.exp_id,
            topic_count=self.topic_count,
            train_option=self.train_option,
        )
        new_population = []
        for point in starting_points_set:
            st_point = point[:12] + [point[15]]
            res = nelder_opt.run_algorithm(
                num_iterations=num_iterations, ini_point=st_point
            )
            solution = list(res["x"])
            solution = (solution[:-1] + point[12:15] + [solution[-1]])  # TODO: check mutation ids
            fitness = -res.fun
            solution_dto = IndividualDTO(
                id=str(uuid.uuid4()),
                data_path=self.data_path,
                dataset=self.dataset,
                params=solution,
                exp_id=self.exp_id,
                alg_id=ALG_ID,
                iteration_id=num_gen,
                topic_count=self.topic_count,
                tag=self.tag,
                fitness_value={AVG_COHERENCE_SCORE: fitness},
                train_option=self.train_option,
            )

            new_population.append(self.ibuilder.make_individual(dto=solution_dto))
        return new_population

    def run(self, verbose=False, visualize_results=False) -> Individual:
        assert self.fitness_estimator.evaluations_counter == 0, \
            "Fitness estimator has non-zero evaluations count and cannot be reused"

        ftime = str(int(time.time()))

        # os.makedirs(LOG_FILE_PATH, exist_ok=True)

        logger.info(f"Starting experiment: {ftime}")

        logger.info(
            f"ALGORITHM PARAMS  number of individuals {self.num_individuals}; "
            f"number of fitness evals "
            f"{self.fitness_estimator.num_fitness_evaluations if self.fitness_estimator.num_fitness_evaluations else 'unlimited'}; "
            f"number of early stopping iterations "
            f"{self.early_stopping_iterations if self.early_stopping_iterations else 'unlimited'}; "
            f"crossover prob {self.elem_cross_prob}"
        )

        population = self.init_population()
        logger.info("POPULATION IS CREATED")

        x, y = [], []
        high_fitness = 0
        best_val_so_far = -10
        early_stopping_counter = 0

        run_id = str(uuid.uuid4())
        for ii in range(self.num_iterations):
            iteration_start_time = time.time()

            logger.info(f"ENTERING GENERATION {ii}")

            self._sort_population(population)
            if self.statistics_collector is not None:
                self.statistics_collector.log_iteration(
                    self.fitness_estimator.evaluations_counter,
                    population[0].fitness_value
                )
            pairs_generator = self.selection(
                population=population,
                best_proc=self.best_proc,
                children_num=self.crossover_children,
            )

            logger.info("PAIRS ARE CREATED")

            # Crossover
            new_generation = self.run_crossover(
                pairs_generator, iteration_num=ii
            )

            self._sort_population(new_generation)
            self._sort_population(population)

            logger.info("CROSSOVER IS OVER")

            if self.use_nelder_mead_in_crossover:
                # TODO: implement Nelder-Mead here
                pass

            del pairs_generator
            gc.collect()

            # population_params = [copy.deepcopy(individ.params) for individ in population]

            the_best_guy_params = copy.deepcopy(population[0].params)
            new_generation = [individ for individ in new_generation if individ.params != the_best_guy_params]

            new_generation_n = min(
                self.num_individuals - int(np.ceil(self.num_individuals * self.best_proc)),
                len(new_generation),
            )
            old_generation_n = self.num_individuals - new_generation_n

            population = population[:old_generation_n] + new_generation[:new_generation_n]

            try:
                del new_generation
            except Exception as e:
                logger.warning(e)
            gc.collect()

            self._sort_population(population)

            self.run_mutation(population)

            # after the mutation we obtain a new population that needs to be evaluated
            for p in population:
                p.dto.iteration_id = ii

            population = self.run_fitness(population, ii)

            # TODO (pipeline) Mutations collection is disabled
            # before_mutation = []  # individual
            # id_mutation = []  # new individual id
            # before_mutation.append(population[i])
            # id_mutation.append(i)
            # for ix, elem in enumerate(before_mutation):
            #     # TODO generation: int, original_params: list, mutated_params: list, original_fitness: float,
            #     #                       mutated_fitness: float
            #     self.metric_collector.save_mutation(
            #         generation=ii,
            #         original_params=elem.params,
            #         mutated_params=population[id_mutation[ix]].params,
            #         original_fitness=elem.fitness_value,
            #         mutated_fitness=population[id_mutation[ix]].fitness_value,
            #     )
            logger.info("MUTATION IS OVER")

            self._sort_population(population)

            if self.use_nelder_mead_in_mutation:
                collected_params = []
                for elem in population:
                    collected_params.append(elem.params)
                random_ids = random.sample(
                    [i for i in range(len(collected_params))], k=3
                )
                starting_points = [collected_params[i] for i in random_ids]

                nm_population = self.apply_nelder_mead(starting_points, num_gen=ii)
                for i, elem in enumerate(nm_population):
                    if population[i].fitness_value < elem.fitness_value:
                        logging.info(
                            f"NM found better solution! {elem.fitness_value} vs {population[i].fitness_value}"
                        )
                        population[i] = elem

            if self.fitness_estimator.num_fitness_evaluations and self.fitness_estimator.evaluations_counter >= self.fitness_estimator.num_fitness_evaluations:
                self.metric_collector.save_fitness(
                    generation=ii,
                    params=[i.params for i in population],
                    fitness=[i.fitness_value for i in population],
                )
                bparams = "".join([str(i) for i in population[0].params])
                logger.info(
                    f"TERMINATION IS TRIGGERED: EVAL NUM (2)."
                    f"DATASET {self.dataset}."
                    f"TOPICS NUM {self.topic_count}."
                    f"RUN ID {run_id}."
                    f"THE BEST FITNESS {population[0].fitness_value}."
                    f"THE BEST PARAMS {bparams}."
                    f"ITERATION TIME {time.time() - iteration_start_time}."
                )
                break

            current_fitness = population[0].fitness_value
            if (current_fitness > high_fitness) or (ii == 0):
                high_fitness = current_fitness

            # if self.surrogate:
            #     if self.calc_scheme == "type1" and not surrogate_iteration:
            #         self.surrogate.fit(
            #             np.array(self.all_params), np.array(self.all_fitness)
            #         )
            #     elif self.calc_scheme == "type2":
            #         self.surrogate.fit(
            #             np.array(self.all_params), np.array(self.all_fitness)
            #         )

            self.fitness_estimator.fit(ii)

            if self.early_stopping_iterations:
                if population[0].fitness_value > best_val_so_far:
                    best_val_so_far = population[0].fitness_value
                    early_stopping_counter = 0
                else:
                    early_stopping_counter += 1
                    if early_stopping_counter == self.early_stopping_iterations:
                        bparams = "".join([str(i) for i in population[0].params])
                        self.metric_collector.save_fitness(
                            generation=ii,
                            params=[i.params for i in population],
                            fitness=[i.fitness_value for i in population],
                        )
                        logger.info(
                            f"TERMINATION IS TRIGGERED: EARLY STOPPING."
                            f"DATASET {self.dataset}."
                            f"TOPICS NUM {self.topic_count}."
                            f"RUN ID {run_id}."
                            f"THE BEST FITNESS {population[0].fitness_value}."
                            f"THE BEST PARAMS {bparams}."
                            f"ITERATION TIME {time.time() - iteration_start_time}."
                        )
                        break

            x.append(ii)
            y.append(population[0].fitness_value)
            self.metric_collector.save_fitness(
                generation=ii,
                params=[i.params for i in population],
                fitness=[i.fitness_value for i in population],
            )
            logger.info(
                f"Population len {len(population)}. "
                f"Best params so far: {population[0].params}, with fitness: {population[0].fitness_value}."
                f"ITERATION TIME: {time.time() - iteration_start_time}"
                f"DATASET {self.dataset}."
                f"TOPICS NUM {self.topic_count}."
                f"RUN ID {run_id}."
            )
            best_solution = population[0]
            self.fitness_estimator.log_best_solution(best_solution, alg_args=" ".join(sys.argv), is_tmp=True)

        if visualize_results:
            self.metric_collector.save_and_visualise_trace()
        else:
            self.metric_collector.save_trace()

        if self.statistics_collector is not None:
            self.statistics_collector.log_iteration(
                self.fitness_estimator.evaluations_counter,
                population[0].fitness_value
            )
        logger.info(f"Y: {y}")
        best_individual = population[0]
        ind = self.fitness_estimator.log_best_solution(best_individual, alg_args=" ".join(sys.argv))
        logger.info(f"Logged the best solution. Obtained fitness is {ind.fitness_value}")

        return ind

    def run_fitness(self, population: List[Individual], ii: int):
        return self.fitness_estimator.estimate(ii,population )

    def run_mutation(self, population):
        for i in range(1, len(population)):
            if random.random() <= population[i].params.mutation_probability:
                params = run_with_retry(
                    action=lambda: population[i].params.mutate(mutation_type=self.mutation_type),
                    condition=lambda candidate: candidate.validate_params()
                )
                if params is None:
                    continue

                dto = IndividualDTO(
                    id=str(uuid.uuid4()),
                    data_path=self.data_path,
                    dataset=self.dataset,
                    params=params,
                    exp_id=self.exp_id,
                    alg_id=ALG_ID,
                    topic_count=self.topic_count,
                    tag=self.tag,
                    train_option=self.train_option,
                )
                population[i] = self.ibuilder.make_individual(dto=dto)


# multistage bag of regularizers approach
# TODO: add penalty for the solution length


class GAmultistage(GA):
    def __init__(self, dataset, num_individuals, max_stages=5):  # max_stage_len
        self.max_stages = max_stages  # amount of unique regularizers
        self.dataset = dataset
        self.bag_of_regularizers = [
            "decor_S",
            "decor_B",
            "S_phi_B",
            "S_phi_S",
            "S_theta_B",
            "S_theta_S",
        ]  # add separate smooth and sparsity
        self.num_individuals = num_individuals
        self.initial_element_stage_probability = 0.5
        self.positioning_matrix = np.full(
            (len(self.bag_of_regularizers), self.max_stages - 1), 0.5
        )
        self.set_regularizer_limits()

    def set_regularizer_limits(
        self,
        low_decor=0,
        high_decor=1e5,
        low_n=1,
        high_n=30,  # minimal value changed to 1
        low_back=0,
        high_back=5,
        low_spb=0,
        high_spb=1e2,
        low_spm=-1e-3,
        high_spm=1e2,
        low_sp_phi=-1e3,
        high_sp_phi=1e3,
    ):
        self.high_decor = high_decor
        self.low_decor = low_decor
        self.low_n = low_n
        self.high_n = high_n
        self.low_back = low_back
        self.high_back = high_back
        self.high_spb = high_spb
        self.low_spb = low_spb
        self.low_spm = low_spm
        self.high_spm = high_spm
        self.low_sp_phi = low_sp_phi
        self.high_sp_phi = high_sp_phi

    # TODO: check if float is needed
    def _init_param(self, param_type):
        if param_type == "decor_S" or param_type == "decor_B":
            return float(
                np.random.uniform(low=self.low_decor, high=self.high_decor, size=1)[0]
            )
        elif param_type == "S_phi_B" or "S_theta_B":
            return float(
                np.random.uniform(low=self.low_spb, high=self.high_spb, size=1)[0]
            )
        elif param_type == "S_phi_S" or "S_theta_S":
            return float(
                np.random.uniform(low=self.low_sp_phi, high=self.high_sp_phi, size=1)[0]
            )
        elif param_type == "n":
            return float(np.random.randint(low=self.low_n, high=self.high_n, size=1)[0])
        elif param_type == "B":
            return float(
                np.random.randint(low=self.low_back, high=self.high_back, size=1)[0]
            )

    def _create_stage(self, stage_num):
        stage_regularizers = []
        for ix, elem in enumerate(self.bag_of_regularizers):
            elem_sample_prob = self.positioning_matrix[ix][stage_num - 1]
            if random.random() < elem_sample_prob:
                stage_regularizers.append(elem)
        stage_regularizers.append("n")
        return stage_regularizers

    def init_individ(self):
        number_of_stages = np.random.randint(low=1, high=self.max_stages, size=1)[0]
        dict_of_stages = [{} for _ in range(number_of_stages)]
        for i in range(number_of_stages):
            regularizers = self._create_stage(i)
            for reg_name in regularizers:
                value = self._init_param(reg_name)
                dict_of_stages[i][reg_name] = value
        dict_of_stages = [{"B": self._init_param("B")}] + dict_of_stages
        return dict_of_stages

    def init_population(self):
        # if random.random() < self.initial_element_stage_probability:
        #
        # for i in range(self.max_stages):
        #     print()

        raise NotImplementedError
