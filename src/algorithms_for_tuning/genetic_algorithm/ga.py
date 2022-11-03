import copy
import gc
import logging
import math
import operator
import os
import random
import sys
import time
import uuid
import warnings
from typing import Optional

import numpy as np
import yaml
from sklearn.ensemble import BaggingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern, WhiteKernel, \
    ConstantKernel, ExpSineSquared, RationalQuadratic
from sklearn.metrics import mean_squared_error
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from yaml import Loader

from algorithms_for_tuning.genetic_algorithm.crossover import crossover
from algorithms_for_tuning.genetic_algorithm.mutation import mutation
from algorithms_for_tuning.genetic_algorithm.selection import selection
from algorithms_for_tuning.individuals import make_individual
from kube_fitness.tasks import IndividualDTO

from algorithms_for_tuning.utils.fitness_estimator import estimate_fitness, prepare_fitness_estimator, log_best_solution

ALG_ID = "ga"
SPEEDUP = True

warnings.filterwarnings("ignore")
logger = logging.getLogger("GA_algo")

# TODO: Add fitness type
def set_surrogate_fitness(value, fitness_type='avg_coherence_score'):
    npmis = {f"npmi_50": None,
             f"npmi_15": None,
             f"npmi_25": None,
             f"npmi_50_list": None}
    scores_dict = {
        fitness_type: value,
        'perplexityScore': None,
        'backgroundTokensRatioScore': None,
        'contrast': None,
        'purity': None,
        'kernelSize': None,
        'npmi_50_list': [None],  # npmi_values_50_list,
        'npmi_50': None,
        'sparsity_phi': None,
        'sparsity_theta': None,
        'topic_significance_uni': None,
        'topic_significance_vacuous': None,
        'topic_significance_back': None,
        'switchP_list': [None],
        'switchP': None,
        'all_topics': None,
        # **coherence_scores,
        **npmis
    }
    return scores_dict




class Surrogate:
    def __init__(self, surrogate_name, **kwargs):
        self.name = surrogate_name
        self.kwargs = kwargs
        self.surrogate = None
        self.br_n_estimators = None
        self.br_n_jobs = None
        self.gpr_kernel = None

    def create(self):
        if self.name == "random-forest-regressor":
            self.surrogate = RandomForestRegressor(**self.kwargs)
        elif self.name == "mlp-regressor":
            if not self.br_n_estimators:
                self.br_n_estimators = self.kwargs['br_n_estimators']
                del self.kwargs['br_n_estimators']
                self.br_n_jobs = self.kwargs['n_jobs']
                del self.kwargs['n_jobs']
                self.kwargs['alpha'] = self.kwargs['mlp_alpha']
                del self.kwargs['mlp_alpha']
            self.surrogate = BaggingRegressor(base_estimator=MLPRegressor(**self.kwargs),
                                              n_estimators=self.br_n_estimators, n_jobs=self.br_n_jobs)
        elif self.name == "GPR":  # tune ??
            if not self.gpr_kernel:
                kernel = self.kwargs['gpr_kernel']
                del self.kwargs['gpr_kernel']
                if kernel == 'RBF':
                    self.gpr_kernel = 1.0 * RBF(1.0)
                elif kernel == "RBFwithConstant":
                    self.gpr_kernel = 1.0 * RBF(1.0) + ConstantKernel()
                elif kernel == 'Matern':
                    self.gpr_kernel = 1.0 * Matern(1.0)
                elif kernel == "WhiteKernel":
                    self.gpr_kernel = 1.0 * WhiteKernel(1.0)
                elif kernel == "ExpSineSquared":
                    self.gpr_kernel = ExpSineSquared()
                elif kernel == "RationalQuadratic":
                    self.gpr_kernel = RationalQuadratic(1.0)
                self.kwargs['kernel'] = self.gpr_kernel
                self.kwargs['alpha'] = self.kwargs['gpr_alpha']
                del self.kwargs['gpr_alpha']
            self.surrogate = GaussianProcessRegressor(**self.kwargs)
        elif self.name == "decision-tree-regressor":
            try:
                if self.kwargs["max_depth"] == 0:
                    self.kwargs["max_depth"] = None
            except KeyError:
                logger.error("No max_depth")
            self.surrogate = DecisionTreeRegressor(**self.kwargs)
        elif self.name == "SVR":
            self.surrogate = SVR(**self.kwargs)
        # else:
        #     raise Exception('Undefined surr')

    def fit(self, X, y):
        logger.debug(f"X: {X}, y: {y}")
        self.create()
        self.surrogate.fit(X, y)

    def score(self, X, y):
        r_2 = self.surrogate.score(X, y)
        y_pred = self.surrogate.predict(X)
        mse = mean_squared_error(y, y_pred)
        rmse = np.sqrt(mse)
        return r_2, mse, rmse

    def predict(self, X):
        m = self.surrogate.predict(X)
        return m


def get_prediction_uncertanty(model, X, surrogate_name, percentile=90):
    interval_len = []
    if surrogate_name == 'random-forest-regressor':
        for x in range(len(X)):
            preds = []
            for pred in model.estimators_:
                prediction = pred.predict(np.array(X[x]).reshape(1, -1))
                preds.append(prediction[0])
            err_down = np.percentile(preds, (100 - percentile) / 2.)
            err_up = np.percentile(preds, 100 - (100 - percentile) / 2.)
            interval_len.append(err_up - err_down)
    elif surrogate_name == 'GPR':
        y_hat, y_sigma = model.predict(X, return_std=True)
        interval_len = list(y_sigma)
    elif surrogate_name == 'decision-tree-regressor':
        raise NotImplementedError
    return interval_len


class GA:
    def __init__(self, dataset, num_individuals, num_iterations,
                 mutation_type='mutation_one_param', crossover_type='blend_crossover',
                 selection_type='fitness_prop', elem_cross_prob=0.2,
                 num_fitness_evaluations: Optional[int] = 500,
                 early_stopping_iterations: Optional[int] = 500,
                 best_proc=0.3, alpha=None, exp_id: Optional[int] = None, surrogate_name=None,
                 calc_scheme='type1', topic_count: Optional[int] = None, tag: Optional[str] = None, **kwargs):

        self.dataset = dataset

        if crossover_type == 'blend_crossover':
            self.crossover_children = 1
        else:
            self.crossover_children = 2

        # if crossover_type == 'blend_crossover':
        #     self.crossover_children = 1
        # else:
        # self.crossover_children = 2
        self.num_individuals = num_individuals
        self.num_iterations = num_iterations
        self.mutation = mutation(mutation_type)
        self.crossover = crossover(crossover_type)
        self.selection = selection(selection_type)
        self.elem_cross_prob = elem_cross_prob
        self.alpha = alpha
        self.evaluations_counter = 0
        self.num_fitness_evaluations = num_fitness_evaluations
        self.early_stopping_iterations = early_stopping_iterations
        self.best_proc = best_proc
        self.all_params = []
        self.all_fitness = []
        if surrogate_name:
            self.surrogate = Surrogate(surrogate_name, **kwargs)
        else:
            self.surrogate = None
        self.exp_id = exp_id
        self.calc_scheme = calc_scheme
        self.topic_count = topic_count
        self.tag = tag
        # params
        self.high_decor = 1e5  # TODO: check param
        self.low_decor = 0
        self.low_n = 0
        self.high_n = 30  # TODO: check param
        self.low_back = 0
        self.high_back = 5
        self.high_spb = 1e2  # TODO: check param
        self.low_spb = 1e-3
        self.low_spm = 1e2
        self.high_spm = -1e-3
        self.low_prob = 0
        self.high_prob = 1

    def init_individ(self, base_model=False):
        val_decor = np.random.uniform(low=self.low_decor, high=self.high_decor, size=1)[0]
        var_n = np.random.randint(low=self.low_n, high=self.high_n, size=4)
        var_back = np.random.randint(low=self.low_back, high=self.high_back, size=1)[0]
        var_sm = np.random.uniform(low=self.low_spb, high=self.high_spb, size=2)
        var_sp = np.random.uniform(low=self.low_spm, high=self.high_spm, size=4)
        ext_mutation_prob = np.random.uniform(low=self.low_prob, high=self.high_prob, size=1)[0]
        ext_elem_mutation_prob = np.random.uniform(low=self.low_prob, high=self.high_prob, size=1)[0]
        ext_mutation_selector = np.random.uniform(low=self.low_prob, high=self.high_prob, size=1)[0]
        val_decor_2 = np.random.uniform(low=self.low_decor, high=self.high_decor, size=1)[0]
        params = [
            val_decor, var_n[0],
            var_sm[0], var_sm[1], var_n[1],
            var_sp[0], var_sp[1], var_n[2],
            var_sp[2], var_sp[3], var_n[3],
            var_back,
            ext_mutation_prob, ext_elem_mutation_prob, ext_mutation_selector,
            val_decor_2
        ]
        if base_model:
            for i in [0, 4, 7, 10, 11, 15]:
                params[i] = 0
        params = [float(i) for i in params]
        return params

    def init_population(self):
        list_of_individuals = []
        for i in range(self.num_individuals):
            if i == 0:
                dto = IndividualDTO(id=str(uuid.uuid4()), dataset=self.dataset, params=self.init_individ(base_model=True),
                                    exp_id=self.exp_id, alg_id=ALG_ID, iteration_id=0,
                                    topic_count=self.topic_count, tag=self.tag)
            else:
                dto = IndividualDTO(id=str(uuid.uuid4()), dataset=self.dataset, params=self.init_individ(),
                                    exp_id=self.exp_id, alg_id=ALG_ID, iteration_id=0,
                                    topic_count=self.topic_count, tag=self.tag)
            # TODO: improve heuristic on search space
            list_of_individuals.append(make_individual(dto=dto))
        population_with_fitness = estimate_fitness(list_of_individuals)
        self.save_params(population_with_fitness)
        if self.surrogate is not None and self.calc_scheme == 'type2':
            self.surrogate.fit(np.array(self.all_params), np.array(self.all_fitness))
            logger.info("Surrogate is initialized!")
        return population_with_fitness

    def _calculate_uncertain_res(self, generation, iteration_num: int, proc=0.3):
        X = np.array([individ.params for individ in generation])
        certanty = get_prediction_uncertanty(self.surrogate.surrogate, X, self.surrogate.name)
        recalculate_num = int(np.floor(len(certanty) * proc))
        logger.info(f'Certanty values: {certanty}')

        certanty, X = (list(t) for t in zip(*sorted(zip(certanty, X.tolist()), reverse=True)))  # check
        calculated = []
        for params in X[:recalculate_num]:
            dto = IndividualDTO(id=str(uuid.uuid4()), params=[float(i) for i in params], dataset=self.dataset,
                                exp_id=self.exp_id, alg_id=ALG_ID, iteration_id=iteration_num,
                                topic_count=self.topic_count, tag=self.tag)
            calculated.append(make_individual(dto=dto))

        calculated = estimate_fitness(calculated)

        self.all_params += [individ.params for individ in calculated]
        self.all_fitness += [individ.fitness_value for individ in calculated]

        pred_y = self.surrogate.predict(X[recalculate_num:])
        for ix, params in enumerate(X[recalculate_num:]):
            dto = IndividualDTO(id=str(uuid.uuid4()), params=params, dataset=self.dataset,
                                fitness_value=pred_y[ix], exp_id=self.exp_id, alg_id=ALG_ID,
                                topic_count=self.topic_count, tag=self.tag)
            calculated.append(make_individual(dto=dto))
        return calculated

    def save_params(self, population):
        params_and_f = [(copy.deepcopy(individ.params), individ.fitness_value) for individ in
                        population if individ.fitness_value not in self.all_fitness]

        def check_val(fval):
            return not (fval is None or math.isnan(fval) or math.isinf(fval))

        def check_params(p):
            return all(check_val(el) for el in p)

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

    def surrogate_calculation(self, population):
        X_val = np.array([copy.deepcopy(individ.params) for individ in population])
        y_pred = self.surrogate.predict(X_val)
        if not SPEEDUP:
            y_val = np.array([individ.fitness_value for individ in population])

            def check_val(fval):
                return not (fval is None or math.isnan(fval) or math.isinf(fval))

            def check_params(p):
                return all(check_val(el) for el in p)

            clean_params_and_f = []
            for i in range(len(y_val)):
                if not check_params(X_val[i]) or not check_val(y_val[i]):
                    logger.warning(f"Bad params or fitness found. Fitness: {y_val[i]}. Params: {X_val[i]}.")
                else:
                    clean_params_and_f.append((X_val[i], y_val[i]))

            X_val = clean_params_and_f[0]
            y_val = clean_params_and_f[1]
            r_2, mse, rmse = self.surrogate.score(X_val, y_val)
            logger.info(f"Real values: {list(y_val)}")
            logger.info(f"Predicted values: {list(y_pred)}")
            logger.info(f"R^2: {r_2}, MSE: {mse}, RMSE: {rmse}")
        for ix, individ in enumerate(population):
            individ.dto.fitness_value = set_surrogate_fitness(y_pred[ix])
        return population

    def _check_param(self, param, bounds):
        param = min(param, bounds[1])
        param = max(param, bounds[0])
        return param

    def check_params_bounds(self, params):
        for i in [2, 3]:
            params[i] = self._check_param(params[i], (self.low_spb, self.high_spb))
        for i in [5, 6, 8, 9]:
            params[i] = self._check_param(params[i], (self.low_spm, self.high_spm))
        for i in [1, 4, 7, 10]:
            params[i] = float(int(params[i]))
            params[i] = self._check_param(params[i], (self.low_n, self.high_n))
        for i in [11]:
            params[i] = float(int(params[i]))
            params[i] = self._check_param(params[i], (self.low_back, self.high_back))
        for i in [12, 13, 14]:
            params[i] = self._check_param(params[i], (self.low_prob, self.high_prob))
        for i in [0, 15]:
            params[i] = self._check_param(params[i], (self.low_decor, self.high_decor))
        return params

    def run_crossover(self, pairs_generator, surrogate_iteration, iteration_num: int):
        new_generation = []

        for i, j in pairs_generator:

            if i is None:
                break

            parent_1 = copy.deepcopy(i.params)
            parent_2 = copy.deepcopy(j.params)

            if self.crossover_children == 2:

                child_1, child_2 = self.crossover(parent_1=parent_1,
                                                  parent_2=parent_2,
                                                  elem_cross_prob=self.elem_cross_prob,
                                                  alpha=self.alpha)

                child_1 = self.check_params_bounds(child_1)
                child_2 = self.check_params_bounds(child_2)

                child1_dto = IndividualDTO(id=str(uuid.uuid4()), dataset=self.dataset, params=child_1,
                                           exp_id=self.exp_id,
                                           alg_id=ALG_ID, iteration_id=iteration_num,
                                           topic_count=self.topic_count, tag=self.tag)
                child2_dto = IndividualDTO(id=str(uuid.uuid4()), dataset=self.dataset, params=child_2,
                                           exp_id=self.exp_id,
                                           alg_id=ALG_ID, iteration_id=iteration_num,
                                           topic_count=self.topic_count, tag=self.tag)

                new_generation.append(make_individual(dto=child1_dto))
                new_generation.append(make_individual(dto=child2_dto))
                self.evaluations_counter += 2
            else:
                child_1 = self.crossover(parent_1=parent_1,
                                         parent_2=parent_2,
                                         elem_cross_prob=self.elem_cross_prob,
                                         alpha=self.alpha
                                         )

                child_1 = self.check_params_bounds(child_1)

                child1_dto = IndividualDTO(id=str(uuid.uuid4()), dataset=self.dataset, params=child_1,
                                           exp_id=self.exp_id,
                                           alg_id=ALG_ID, iteration_id=iteration_num,
                                           topic_count=self.topic_count, tag=self.tag)
                new_generation.append(make_individual(dto=child1_dto))

                self.evaluations_counter += 1

        logger.info(f"CURRENT COUNTER: {self.evaluations_counter}")

        if len(new_generation) > 0:

            fitness_calc_time_start = time.time()
            if not SPEEDUP or not self.surrogate:
                new_generation = estimate_fitness(new_generation)
                self.save_params(new_generation)
            logger.info(f"ize of the new generation is {len(new_generation)}")
            logger.info(f"TIME OF THE FITNESS FUNCTION IN CROSSOVER: {time.time() - fitness_calc_time_start}")

            if self.surrogate:
                if self.calc_scheme == 'type1':
                    if surrogate_iteration:
                        self.surrogate_calculation(new_generation)
                    elif not surrogate_iteration and SPEEDUP:
                        new_generation = estimate_fitness(new_generation)
                        self.save_params(new_generation)
                elif self.calc_scheme == 'type2':
                    new_generation = self._calculate_uncertain_res(new_generation, iteration_num=iteration_num)
                    self.save_params(new_generation)

        return new_generation

    def run(self, verbose=False):
        prepare_fitness_estimator()

        self.evaluations_counter = 0
        ftime = str(int(time.time()))

        # os.makedirs(LOG_FILE_PATH, exist_ok=True)

        logger.info(f"Starting experiment: {ftime}")

        logger.info(f"ALGORITHM PARAMS  number of individuals {self.num_individuals}; "
                    f"number of fitness evals "
                    f"{self.num_fitness_evaluations if self.num_fitness_evaluations else 'unlimited'}; "
                    f"number of early stopping iterations "
                    f"{self.early_stopping_iterations if self.early_stopping_iterations else 'unlimited'}; "
                    f"crossover prob {self.elem_cross_prob}")

        # population initialization
        population = self.init_population()

        self.evaluations_counter = self.num_individuals

        logger.info("POPULATION IS CREATED")

        x, y = [], []
        high_fitness = 0
        surrogate_iteration = False
        best_val_so_far = -10
        early_stopping_counter = 0

        run_id = str(uuid.uuid4())

        for ii in range(self.num_iterations):
            iteration_start_time = time.time()

            logger.info(f"ENTERING GENERATION {ii}")

            if self.surrogate is not None:
                surrogate_iteration = False
                if ii % 2 != 0:
                    surrogate_iteration = True

            population.sort(key=operator.attrgetter('fitness_value'), reverse=True)
            pairs_generator = self.selection(population=population,
                                             best_proc=self.best_proc,
                                             children_num=self.crossover_children)

            logger.info(f"PAIRS ARE CREATED")

            # Crossover
            new_generation = self.run_crossover(pairs_generator, surrogate_iteration, iteration_num=ii)

            new_generation.sort(key=operator.attrgetter('fitness_value'), reverse=True)
            population.sort(key=operator.attrgetter('fitness_value'), reverse=True)

            logger.info("CROSSOVER IS OVER")

            if self.num_fitness_evaluations and self.evaluations_counter >= self.num_fitness_evaluations:
                bparams = ''.join([str(i) for i in population[0].params])
                logger.info(f"TERMINATION IS TRIGGERED: EVAL NUM."
                            f"DATASET {self.dataset}."
                            f"TOPICS NUM {self.topic_count}."
                            f"RUN ID {run_id}."
                            f"THE BEST FITNESS {population[0].fitness_value}."
                            f"THE BEST PARAMS {bparams}."
                            f"ITERATION TIME {time.time() - iteration_start_time}.")
                # return population[0].fitness_value
                break

            del pairs_generator
            gc.collect()

            # population_params = [copy.deepcopy(individ.params) for individ in population]

            the_best_guy_params = copy.deepcopy(population[0].params)
            new_generation = [individ for individ in new_generation if individ.params != the_best_guy_params]

            new_generation_n = min((self.num_individuals - int(np.ceil(self.num_individuals * self.best_proc))),
                                   len(new_generation))
            old_generation_n = self.num_individuals - new_generation_n

            population = population[:old_generation_n] + new_generation[:new_generation_n]

            try:
                del new_generation
            except:
                pass

            gc.collect()

            population.sort(key=operator.attrgetter('fitness_value'), reverse=True)

            try:
                del population[self.num_individuals]
            except:
                pass

            # mutation params 12, 13
            # TODO: check this code
            for i in range(1, len(population)):

            #     if random.random() <= population[i].params[12]:
            #         for idx in range(3):
            #             if random.random() < population[i].params[13]:
            #                 if idx == 0:
            #                     population[i].params[12] = np.random.uniform(low=self.low_prob,
            #                                                                  high=self.high_prob, size=1)[0]
            #                 elif idx == 1:
            #                     population[i].params[13] = np.random.uniform(low=self.low_prob,
            #                                                                  high=self.high_prob, size=1)[0]
            #                 elif idx == 2:
            #                     population[i].params[13] = np.random.uniform(low=self.low_prob,
            #                                                                  high=self.high_prob, size=1)[0]

                if random.random() <= population[i].params[12]:
                    params = self.mutation(copy.deepcopy(population[i].params),
                                           elem_mutation_prob=copy.deepcopy(population[i].params[13]),
                                           low_spb=self.low_spb, high_spb=self.high_spb,
                                           low_spm=self.low_spm, high_spm=self.high_spm,
                                           low_n=self.low_n, high_n=self.high_n,
                                           low_back=self.low_back, high_back=self.high_back,
                                           low_decor=self.low_decor, high_decor=self.high_decor
                                           )

                    fix_value_13 = population[i].params[13]
                    for ix in [12, 13, 14]:
                        if random.random() < fix_value_13:
                            population[i].params[12] = np.random.uniform(low=self.low_prob,
                                                                         high=self.high_prob, size=1)[0]
                    params = self.check_params_bounds(params)
                    dto = IndividualDTO(id=str(uuid.uuid4()), dataset=self.dataset,
                                        params=[float(i) for i in params],
                                        exp_id=self.exp_id, alg_id=ALG_ID,
                                        topic_count=self.topic_count, tag=self.tag)
                    population[i] = make_individual(dto=dto)
                self.evaluations_counter += 1


            # after the mutation we obtain a new population that needs to be evaluated
            for p in population:
                p.dto.iteration_id = ii

            fitness_calc_time_start = time.time()
            if not SPEEDUP or not self.surrogate:
                population = estimate_fitness(population)
                self.save_params(population)
            logger.info(f"TIME OF THE FITNESS FUNCTION IN MUTATION: {time.time() - fitness_calc_time_start}")

            if self.calc_scheme == 'type1' and self.surrogate:
                if surrogate_iteration and self.surrogate:
                    self.surrogate_calculation(population)
                elif not surrogate_iteration and SPEEDUP and self.surrogate:
                    population = estimate_fitness(population)
                    self.save_params(population)
            elif self.calc_scheme == 'type2' and self.surrogate:
                population = self._calculate_uncertain_res(population, iteration_num=ii)
                self.save_params(population)

            ###
            logger.info("MUTATION IS OVER")

            population.sort(key=operator.attrgetter('fitness_value'), reverse=True)

            if self.num_fitness_evaluations and self.evaluations_counter >= self.num_fitness_evaluations:
                bparams = ''.join([str(i) for i in population[0].params])
                logger.info(f"TERMINATION IS TRIGGERED: EVAL NUM (2)."
                            f"DATASET {self.dataset}."
                            f"TOPICS NUM {self.topic_count}."
                            f"RUN ID {run_id}."
                            f"THE BEST FITNESS {population[0].fitness_value}."
                            f"THE BEST PARAMS {bparams}."
                            f"ITERATION TIME {time.time() - iteration_start_time}.")
                # return population[0].fitness_value
                break

            current_fitness = population[0].fitness_value
            if (current_fitness > high_fitness) or (ii == 0):
                high_fitness = current_fitness

            if self.surrogate:
                if self.calc_scheme == 'type1' and not surrogate_iteration:
                    self.surrogate.fit(np.array(self.all_params), np.array(self.all_fitness))
                elif self.calc_scheme == 'type2':
                    self.surrogate.fit(np.array(self.all_params), np.array(self.all_fitness))

            if self.early_stopping_iterations:
                if population[0].fitness_value > best_val_so_far:
                    best_val_so_far = population[0].fitness_value
                    early_stopping_counter = 0
                else:
                    early_stopping_counter += 1
                    if early_stopping_counter == self.early_stopping_iterations:
                        bparams = ''.join([str(i) for i in population[0].params])
                        logger.info(f"TERMINATION IS TRIGGERED: EARLY STOPPING."
                                    f"DATASET {self.dataset}."
                                    f"TOPICS NUM {self.topic_count}."
                                    f"RUN ID {run_id}."
                                    f"THE BEST FITNESS {population[0].fitness_value}."
                                    f"THE BEST PARAMS {bparams}."
                                    f"ITERATION TIME {time.time() - iteration_start_time}.")
                        break

            bparams = ''.join([str(i) for i in population[0].params])
            x.append(ii)
            y.append(population[0].fitness_value)
            logger.info(f"Population len {len(population)}. "
                        f"Best params so far: {population[0].params}, with fitness: {population[0].fitness_value}." 
                        f"ITERATION TIME: {time.time() - iteration_start_time}"
                        f"DATASET {self.dataset}."
                        f"TOPICS NUM {self.topic_count}."
                        f"RUN ID {run_id}.")

        logger.info(f"Y: {y}")

        best_individual = population[0]
        log_best_solution(best_individual, alg_args=' '.join(sys.argv))

        return best_individual.fitness_value
