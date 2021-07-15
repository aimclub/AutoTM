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

from mutation import mutation
from crossover import crossover
from selection import selection
import random
import logging
import uuid

from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import BaggingRegressor
from sklearn.gaussian_process.kernels import RBF

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


class surrogate:
    def __init__(self, surrogate_name, random_state=None):
        self.name = surrogate_name
        self.random_state = random_state
        self.surrogate = None

    def create(self):
        if self.name == "random-forest-regressor":
            self.surrogate = RandomForestRegressor(random_state=self.random_state)
        elif self.name == "mlp-regressor":
            self.surrogate = BaggingRegressor(base_estimator=MLPRegressor(activation='tanh', alpha=0.001,
                                                                          early_stopping=True,
                                                                          hidden_layer_sizes=(10, 20, 10),
                                                                          solver='lbfgs'), n_estimators=5,
                                              random_state=self.random_state)
        elif self.name == "GPR":  # tune ??
            kernel = RBF()
            self.surrogate = GaussianProcessRegressor(kernel=kernel, random_state=self.random_state)
        elif self.name == "decision-tree-regressor":
            self.surrogate = DecisionTreeRegressor()

    #             kernel =
    #             self.regr = GaussianProcessRegressor()

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
                 best_proc=0.3, alpha=None, exp_id: Optional[int] = None):
        self.dataset = dataset

        if crossover_type == 'blend_crossover':
            self.crossover_children = 1
        else:
            self.crossover_children = 2

        self.num_individuals = num_individuals
        self.num_iterations = num_iterations
        self.mutation = mutation(mutation_type)  # mutation function
        self.crossover = crossover(crossover_type)  # crossover finction
        self.selection = selection(selection_type)  # selection function
        self.elem_cross_prob = elem_cross_prob
        self.alpha = alpha
        self.num_fitness_evaluations = num_fitness_evaluations
        self.best_proc = best_proc

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

    def run(self, verbose=False):
        prepare_fitness_estimator()

        evaluations_counter = 0
        ftime = str(int(time.time()))

        # os.makedirs(LOG_FILE_PATH, exist_ok=True)

        logger.info(f"Starting experiment: {ftime}")

        logger.info(f"ALGORITHM PARAMS  number of individuals {self.num_individuals}; "
                    f"number of fitness evals {self.num_fitness_evaluations}; "
                    f"crossover prob {self.elem_cross_prob}")

        # population initialization
        population = self.init_population()

        evaluations_counter = self.num_individuals

        logger.info("POPULATION IS CREATED")

        x, y = [], []
        high_fitness = 0
        best_solution = None
        for ii in range(self.num_iterations):

            new_generation = []

            logger.info(f"ENTERING GENERATION {ii}")

            population.sort(key=operator.attrgetter('fitness_value'), reverse=True)
            pairs_generator = self.selection(population=population,
                                             best_proc=self.best_proc, children_num=self.crossover_children)

            logger.info(f"PAIRS ARE CREATED")

            # Crossover
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
                    evaluations_counter += 2
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

                    evaluations_counter += 1

            logger.info(f"CURRENT COUNTER: {evaluations_counter}")

            new_generation = estimate_fitness(new_generation)

            new_generation.sort(key=operator.attrgetter('fitness_value'), reverse=True)
            population.sort(key=operator.attrgetter('fitness_value'), reverse=True)

            logger.info("CROSSOVER IS OVER")

            if evaluations_counter >= self.num_fitness_evaluations:
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
                                                  alg_id=ALG_ID)  # TODO: check mutation
                evaluations_counter += 1

            population = estimate_fitness(population)

            ###
            logger.info("MUTATION IS OVER")

            population.sort(key=operator.attrgetter('fitness_value'), reverse=True)

            if evaluations_counter >= self.num_fitness_evaluations:
                bparams = ''.join([str(i) for i in population[0].params])
                logger.info(f"TERMINATION IS TRIGGERED."
                            f"THE BEST FITNESS {population[0].fitness_value}."
                            f"THE BEST PARAMS {bparams}.")
                # return population[0].fitness_value
                break

            current_fitness = population[0].fitness_value
            if (current_fitness > high_fitness) or (ii == 0):
                high_fitness = current_fitness
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
