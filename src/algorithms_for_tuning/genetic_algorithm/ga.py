import os
import gc
import yaml
from yaml import Loader
import operator
import warnings
import numpy as np
import time
from typing import List, Optional
import copy

from src.algorithms_for_tuning.genetic_algorithm.mutation import mutation
from src.algorithms_for_tuning.genetic_algorithm.crossover import crossover
from src.algorithms_for_tuning.genetic_algorithm.selection import selection
import random
import logging
import uuid

from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import BaggingRegressor
from sklearn.gaussian_process.kernels import RBF, Matern, WhiteKernel, \
    ConstantKernel, ExpSineSquared, RationalQuadratic

from sklearn.metrics import mean_squared_error

ALG_ID = "ga"

warnings.filterwarnings("ignore")
logger = logging.getLogger("GA_algo")

from kube_fitness.tasks import IndividualDTO, TqdmToLogger

if "FITNESS_CONFIG_PATH" in os.environ:
    filepath = os.environ["FITNESS_CONFIG_PATH"]
else:
    filepath = "../../algorithms_for_tuning/genetic_algorithm/config.yaml"

with open(filepath, "r") as file:
    config = yaml.load(file, Loader=Loader)

if not config['testMode']:
    from kube_fitness.tasks import make_celery_app as prepare_fitness_estimator
    from kube_fitness.tasks import parallel_fitness as estimate_fitness
    from kube_fitness.tasks import log_best_solution
else:
    from tqdm import tqdm


    def prepare_fitness_estimator():
        pass


    def estimate_fitness(population: List[IndividualDTO],
                         use_tqdm: bool = False,
                         tqdm_check_period: int = 2) -> List[IndividualDTO]:
        results = []

        tqdm_out = TqdmToLogger(logger, level=logging.INFO)
        for p in tqdm(population, file=tqdm_out):
            individual = copy.deepcopy(p)
            individual.fitness_value = random.random()
            results.append(individual)

        return results


    def log_best_solution(individual: IndividualDTO):
        pass


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

    def fit(self, X, y):
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


class GA:
    def __init__(self, dataset, num_individuals, num_iterations,
                 mutation_type='mutation_one_param', crossover_type='blend_crossover',
                 selection_type='fitness_prop', elem_cross_prob=0.2, num_fitness_evaluations=200,
                 best_proc=0.3, alpha=None, exp_id: Optional[int] = None, surrogate_name=None,
                 **kwargs):
        self.dataset = dataset

        if crossover_type == 'blend_crossover':
            self.crossover_children = 1
        else:
            self.crossover_children = 2

        self.num_individuals = num_individuals
        self.num_iterations = num_iterations
        self.mutation = mutation(mutation_type)
        self.crossover = crossover(crossover_type)
        self.selection = selection(selection_type)
        self.elem_cross_prob = elem_cross_prob
        self.alpha = alpha
        self.evaluations_counter = 0
        self.num_fitness_evaluations = num_fitness_evaluations
        self.best_proc = best_proc
        self.current_surrogate = None
        self.all_params = []
        self.all_fitness = []
        if surrogate_name:
            self.surrogate = Surrogate(surrogate_name, **kwargs)
        else:
            self.surrogate = None
        self.exp_id = exp_id

    @staticmethod
    def init_individ(high_decor=1e5,
                     high_n=8, high_spb=1e2,
                     low_spm=-1e2):
        val_decor = np.random.uniform(low=0, high=high_decor, size=1)[0]
        var_n = np.random.randint(low=0, high=high_n, size=5)
        var_sm = np.random.uniform(low=1e-3, high=high_spb, size=2)
        var_sp = np.random.uniform(low=low_spm, high=-1e-3, size=4)
        ext_mutation_prob = np.random.uniform(low=0, high=1, size=1)[0]
        ext_elem_mutation_prob = np.random.uniform(low=0, high=1, size=1)[0]
        ext_mutation_selector = np.random.uniform(low=0, high=1, size=1)[0]
        val_decor_2 = np.random.uniform(low=0, high=high_decor, size=1)[0]
        params = [
            val_decor, var_n[0],
            var_sm[0], var_sm[1], var_n[1],
            var_sp[0], var_sp[1], var_n[2],
            var_sp[2], var_sp[3], var_n[3],
            var_n[4],
            ext_mutation_prob, ext_elem_mutation_prob, ext_mutation_selector,
            val_decor_2
        ]
        params = [float(i) for i in params]
        return params

    def init_population(self):
        list_of_individuals = []
        for i in range(self.num_individuals):
            list_of_individuals.append(IndividualDTO(id=str(uuid.uuid4()),
                                                     dataset=self.dataset,
                                                     params=self.init_individ(),
                                                     exp_id=self.exp_id,
                                                     alg_id=ALG_ID))
        population_with_fitness = estimate_fitness(list_of_individuals)
        return population_with_fitness

    def save_params(self, population):
        self.all_params += [individ.params for individ in population]
        self.all_fitness += [individ.fitness_value for individ in population]

    # TODO: check if this method should return something
    def surrogate_calculation(self, population):
        X_val = np.array([individ.params for individ in population])
        y_val = np.array([individ.fitness_value for individ in population])
        y_pred = self.surrogate.predict(X_val)
        r_2, mse, rmse = self.surrogate.score(X_val, y_val)
        logger.info("Real values: ", list(y_val))
        logger.info("Predicted values: ", list(y_pred))
        logger.info(f"R^2: {r_2}, MSE: {mse}, RMSE: {rmse}")
        for ix, individ in enumerate(population):
            individ.fitness_value = y_pred[ix]

    def run_crossover(self, pairs_generator, surrogate_iteration):
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

                new_generation.append(IndividualDTO(id=str(uuid.uuid4()),
                                                    dataset=self.dataset,
                                                    params=child_1,
                                                    exp_id=self.exp_id,
                                                    alg_id=ALG_ID))
                new_generation.append(IndividualDTO(id=str(uuid.uuid4()),
                                                    dataset=self.dataset,
                                                    params=child_2,
                                                    exp_id=self.exp_id,
                                                    alg_id=ALG_ID))
                self.evaluations_counter += 2
            else:
                child_1 = self.crossover(parent_1=parent_1,
                                         parent_2=parent_2,
                                         elem_cross_prob=self.elem_cross_prob,
                                         alpha=self.alpha
                                         )
                new_generation.append(IndividualDTO(id=str(uuid.uuid4()),
                                                    dataset=self.dataset,
                                                    params=child_1,
                                                    exp_id=self.exp_id,
                                                    alg_id=ALG_ID))

                self.evaluations_counter += 1

        logger.info(f"CURRENT COUNTER: {self.evaluations_counter}")
        fitness_calc_time_start = time.time()
        new_generation = estimate_fitness(new_generation)
        logger.info(f"TIME OF THE FITNESS FUNCTION IN CROSSOVER: {time.time() - fitness_calc_time_start}")
        if surrogate_iteration:
            self.surrogate_calculation(new_generation)
        else:
            self.save_params(new_generation)

        return new_generation

    def run(self, verbose=False):
        prepare_fitness_estimator()

        self.evaluations_counter = 0
        ftime = str(int(time.time()))

        # os.makedirs(LOG_FILE_PATH, exist_ok=True)

        logger.info(f"Starting experiment: {ftime}")

        logger.info(f"ALGORITHM PARAMS  number of individuals {self.num_individuals}; "
                    f"number of fitness evals {self.num_fitness_evaluations}; "
                    f"crossover prob {self.elem_cross_prob}")

        # population initialization
        population = self.init_population()

        self.evaluations_counter = self.num_individuals

        logger.info("POPULATION IS CREATED")

        x, y = [], []
        high_fitness = 0
        surrogate_iteration = False

        for ii in range(self.num_iterations):

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
            new_generation = self.run_crossover(pairs_generator, surrogate_iteration)

            new_generation.sort(key=operator.attrgetter('fitness_value'), reverse=True)
            population.sort(key=operator.attrgetter('fitness_value'), reverse=True)

            logger.info("CROSSOVER IS OVER")

            if self.evaluations_counter >= self.num_fitness_evaluations:
                bparams = ''.join([str(i) for i in population[0].params])
                logger.info(f"TERMINATION IS TRIGGERED."
                            f"THE BEST FITNESS {population[0].fitness_value}."
                            f"THE BEST PARAMS {bparams}.")
                # return population[0].fitness_value
                break

            del pairs_generator
            gc.collect()

            population_params = [copy.deepcopy(individ.params) for individ in population]

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
            for i in range(1, len(population)):
                if random.random() <= population[i].params[12]:
                    for idx in range(3):
                        if random.random() < population[i].params[13]:
                            if idx == 0:
                                population[i].params[12] = np.random.uniform(low=0, high=1, size=1)[0]
                            elif idx == 2:
                                population[i].params[13] = np.random.uniform(low=0, high=1, size=1)[0]
                            elif idx == 3:
                                population[i].params[13] = np.random.uniform(low=0, high=1, size=1)[0]

                if random.random() <= population[i].params[12]:
                    params = self.mutation(copy.deepcopy(population[i].params),
                                           elem_mutation_prob=copy.deepcopy(population[i].params[13]))
                    population[i] = IndividualDTO(id=str(uuid.uuid4()),
                                                  dataset=self.dataset,
                                                  params=[float(i) for i in params],
                                                  exp_id=self.exp_id,
                                                  alg_id=ALG_ID)
                self.evaluations_counter += 1

            fitness_calc_time_start = time.time()
            population = estimate_fitness(population)
            logger.info(f"TIME OF THE FITNESS FUNCTION IN MUTATION: {time.time() - fitness_calc_time_start}")

            if surrogate_iteration:
                self.surrogate_calculation(population)
            else:
                self.save_params(population)

            ###
            logger.info("MUTATION IS OVER")

            population.sort(key=operator.attrgetter('fitness_value'), reverse=True)

            if self.evaluations_counter >= self.num_fitness_evaluations:
                bparams = ''.join([str(i) for i in population[0].params])
                logger.info(f"TERMINATION IS TRIGGERED."
                            f"THE BEST FITNESS {population[0].fitness_value}."
                            f"THE BEST PARAMS {bparams}.")
                # return population[0].fitness_value
                break

            current_fitness = population[0].fitness_value
            if (current_fitness > high_fitness) or (ii == 0):
                high_fitness = current_fitness

            if not surrogate_iteration:
                self.surrogate.fit(np.array(self.all_params), np.array(self.all_fitness))

            bparams = ''.join([str(i) for i in population[0].params])
            logger.info(f"TERMINATION IS TRIGGERED."
                        f"THE BEST FITNESS {population[0].fitness_value}."
                        f"THE BEST PARAMS {bparams}.")
            x.append(ii)
            y.append(population[0].fitness_value)
            logger.info(f"Population len {len(population)}. "
                        f"Best params so far: {population[0].params}, with fitness: {population[0].fitness_value}.")

        logger.info(f"Y: {y}")

        best_individual = population[0]
        log_best_solution(best_individual)

        return best_individual.fitness_value
