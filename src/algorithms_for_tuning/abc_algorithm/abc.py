#!/usr/bin/env python3
import copy
import logging
import os
import sys
import uuid
import warnings
from logging import config
from typing import List

# from kube_fitness.tasks import make_celery_app, parallel_fitness, IndividualDTO
# from kube_fitness.tasks import IndividualDTO, TqdmToLogger
import click
import numpy as np
import yaml
from numpy import random
from numpy.random import permutation, rand
from sklearn.preprocessing import MinMaxScaler
from yaml import Loader

from algorithms_for_tuning.utils import make_log_config_dict

warnings.filterwarnings("ignore")

from kube_fitness.tasks import IndividualDTO, TqdmToLogger

logger = logging.getLogger("ABC")

# getting config vars
if "FITNESS_CONFIG_PATH" in os.environ:
    filepath = os.environ["FITNESS_CONFIG_PATH"]
else:
    filepath = "../../algorithms_for_tuning/abc_algorithm/config.yaml"

with open(filepath, "r") as file:
    config = yaml.load(file, Loader=Loader)

if not config['testMode']:
    from kube_fitness.tasks import parallel_fitness as estimate_fitness
else:
    # from kube_fitness.tm import calculate_fitness_of_individual, TopicModelFactory
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

NUM_FITNESS_EVALUATIONS = config['globalAlgoParams']['numEvals']
PROBLEM_DIM = config['globalAlgoParams']['problemDim']


@click.command(context_settings=dict(allow_extra_args=True))
@click.option('--dataset', help='dataset name in the config')
@click.option('--num-individuals', default=10, help='colony size')  # colony size
@click.option('--max-num-trials', default=5, help='maximum number of source trials')
@click.option('--init-method', default='latin_hypercube',
              help='method of population initialization (latin hypercube or random)')
@click.option('--log-file', default="/var/log/tm-alg.log",
              help='a log file to write logs of the algorithm execution to')
@click.option('--exp-id', type=int, help='mlflow experiment id')
def run_algorithm(dataset, num_individuals,
                  max_num_trials, init_method,
                  log_file, exp_id):
    run_uid = uuid.uuid4() if not config['testMode'] else None
    logging_config = make_log_config_dict(filename=log_file, uid=run_uid)
    logging.config.dictConfig(logging_config)

    logger.info(f"Starting a new run of algorithm. Args: {sys.argv[1:]}")
    abc_algo = ABC(dataset=dataset,
                   colony_size=num_individuals,
                   max_num_trials=max_num_trials,
                   init_method=init_method,
                   problem_dim=PROBLEM_DIM,
                   num_fitness_evaluations=NUM_FITNESS_EVALUATIONS,
                   exp_id=exp_id)
    abc_algo.run(16)
    print(round(abc_algo.best_solution - 1, 3) * (-1))  # according to the code logic


def lhd(n_sam, n_val, val_rng=None, method='random', criterion=None,
        iterations=1000, get_score=False, quickscan=True, constraints=None):
    # Make sure that if val_rng is given, that it is valid
    if val_rng is not None:
        # If val_rng is 1D, convert it to 2D (expected for 'n_val' = 1)
        val_rng = np.array(val_rng, ndmin=2)

        # Check if the given val_rng is in the correct shape
        if not (val_rng.shape == (n_val, 2)):
            raise Exception("'val_rng' has incompatible shape: %s != (%s, %s)" % (val_rng.shape, n_val, 2))

    # TODO: Implement constraints method again!
    # Make sure that constraints is a numpy array
    if constraints is not None:
        constraints = np.array(constraints, ndmin=2)

    if constraints is None:
        pass
    elif constraints.shape[-1] == 0:
        # If constraints is empty, there are no constraints
        constraints = None
    elif constraints.ndim != 2:
        # If constraints is not two-dimensional, it is invalid
        raise Exception("Constraints must be two-dimensional!")
    elif constraints.shape[-1] == n_val:
        # If constraints has the same number of values, it is valid
        # constraints = _extract_sam_set(constraints, val_rng)
        raise NotImplementedError("You should not get here")
    else:
        # If not empty and not right shape, it is invalid
        raise Exception("Constraints has incompatible number of values: "
                         "%s =! %s" % (np.shape(constraints)[1], n_val))

    # Check for cases in which some methods make no sense
    if n_sam == 1 and method.lower() in ('fixed', 'f'):
        method = 'center'
    elif criterion is not None and method.lower() in ('random', 'r'):
        method = 'fixed'

    # Check for cases in which some criterions make no sense
    # If so, criterion will be changed to something useful
    if criterion is None:
        pass
    elif n_sam == 1:
        criterion = None
    elif n_val == 1 or n_sam == 2:
        criterion = None
    elif isinstance(criterion, (int, float)):
        if not (0 <= criterion <= 1):
            raise ValueError("Input argument 'criterion' can only have a "
                             "normalized value as a float value!")
    elif criterion.lower() not in ('maximin', 'correlation', 'multi'):
        raise ValueError("Input argument 'criterion' can only have {'maximin',"
                         " 'correlation', 'multi'} as string values!")

    # Pick correct lhs-method according to method
    if method.lower() in ('random', 'r'):
        sam_set = _lhd_random(n_sam, n_val)
    elif method.lower() in ('fixed', 'f'):
        sam_set = _lhd_fixed(n_sam, n_val)
    elif method.lower() in ('center', 'c'):
        sam_set = _lhd_center(n_sam, n_val)
    else:
        raise Exception(f"Method is not known: {method.lower()}")

    # Pick correct criterion
    if criterion is not None:
        raise NotImplementedError("You should not get here")
        # multi_obj = Multi_LHD(sam_set, criterion, iterations, quickscan,
        #                       constraints)
        # sam_set, mm_val, corr_val, multi_val = multi_obj()

    # If a val_rng was given, scale sam_set to this range
    if val_rng is not None:
        # Scale sam_set according to val_rng
        sam_set = val_rng[:, 0] + sam_set * (val_rng[:, 1] - val_rng[:, 0])

    if get_score and criterion is not None:
        raise NotImplementedError("You should not get here")
        # return (sam_set, np.array([mm_val, corr_val, multi_val]))
    else:
        return (sam_set)


def _lhd_random(n_sam, n_val):
    # Generate the equally spaced intervals/bins
    bins = np.linspace(0, 1, n_sam + 1)

    # Obtain lower and upper bounds of bins
    bins_low = bins[0:n_sam]
    bins_high = bins[1:n_sam + 1]

    # Pair values randomly together to obtain random samples
    sam_set = np.zeros([n_sam, n_val])
    for i in range(n_val):
        sam_set[:, i] = permutation(bins_low + rand(n_sam) * (bins_high - bins_low))

    # Return sam_set
    return (sam_set)


def _lhd_fixed(n_sam, n_val):
    # Generate the maximally spaced values in every dimension
    val = np.linspace(0, 1, n_sam)

    # Pair values randomly together to obtain random samples
    sam_set = np.zeros([n_sam, n_val])
    for i in range(n_val):
        sam_set[:, i] = permutation(val)

    # Return sam_set
    return (sam_set)


def _lhd_center(n_sam, n_val):
    # Generate the equally spaced intervals/bins
    bins = np.linspace(0, 1, n_sam + 1)

    # Obtain lower and upper bounds of bins
    bins_low = bins[0:n_sam]
    bins_high = bins[1:n_sam + 1]

    # Capture centers of every bin
    center_num = (bins_low + bins_high) / 2

    # Pair values randomly together to obtain random samples
    sam_set = np.zeros([n_sam, n_val])
    for i in range(n_val):
        sam_set[:, i] = permutation(center_num)

    # Return sam_set
    return (sam_set)


def abc_fitness(sources):
    for source in sources:
        source.fitness_value += 1


class ABC:
    def __init__(self,
                 dataset,
                 colony_size,
                 max_num_trials=5,
                 init_method='latin_hypercube',
                 problem_dim=16,
                 num_fitness_evaluations=150,
                 exp_id=3
                 ):
        self.exp_id = exp_id
        self.dataset = dataset
        self.colony_size = colony_size
        self.problem_dim = problem_dim
        self.scout_limit = (colony_size * problem_dim) / 2  # TODO: fix
        self.food_resources_num = int(colony_size / 2)
        self.employed_bees = None
        self.scout_bees = []
        self.init_resources_method = self.init_food_sources_method(init_method)
        self.trials = [0 for _ in range(self.food_resources_num)]
        self.probabilities = [0 for _ in range(self.food_resources_num)]
        self.max_num_trials = max_num_trials  # TODO: check the vals
        self.best_solution = -1
        self.fitness_evals = 0
        self.num_fitness_evaluations = num_fitness_evaluations

        # variable limits
        self.low_decor = 0
        self.high_decor = 1e5
        self.low_n = 0
        self.high_n = 5  # changed from 8 to 15
        self.low_sm = 1e-3
        self.high_sm = 1e2
        self.low_sp = -1e2
        self.high_sp = -1e-3
        self.low_back = 0
        self.high_back = 8
        self.low_mutation_prob = 0  # TODO: check low and high vals
        self.high_mutation_prob = 1
        self.low_elem_mutation_prob = 0
        self.high_elem_mutation_prob = 1
        self.low_ext_mutation_selector_prob = 0
        self.high_ext_mutation_selector_prob = 1
        self.params_limits = [
            (self.low_decor, self.high_decor),
            (self.low_n, self.high_n),
            (self.low_sm, self.high_sm),
            (self.low_sm, self.high_sm),
            (self.low_n, self.high_n),
            (self.low_sp, self.high_sp),
            (self.low_sp, self.high_sp),
            (self.low_n, self.high_n),
            (self.low_sp, self.high_sp),
            (self.low_sp, self.high_sp),
            (self.low_n, self.high_n),
            (self.low_back, self.high_back),
            #             (self.low_mutation_prob, self.high_mutation_prob),
            #             (self.low_elem_mutation_prob, self.high_elem_mutation_prob),
            #             (self.low_ext_mutation_selector_prob, self.high_ext_mutation_selector_prob),
            (0, 1)
        ]

    def init_food_sources_method(self, method):
        if method == 'latin_hypercube':
            return self._latin_hypercube_init_food
        elif method == 'random':
            return self._random_init_food

    def _latin_hypercube_init_food(self,
                                   resources_num=None):
        if resources_num is None:
            resources_num = self.food_resources_num
        cube_params = lhd(resources_num, self.problem_dim - 3, method='fixed')
        #         cube_params = np.concatenate([cube_params[:12], np.array([0, 0, 0]), cube_params[-1]])
        for ix, param_lim in enumerate(self.params_limits):
            scaler = MinMaxScaler(param_lim)
            cube_params[:, ix] = scaler.fit_transform(
                cube_params[:, ix].reshape(-1, 1)).reshape(1, -1)[0]
        list_of_individuals = []
        for row in cube_params:
            row = row.tolist()
            row = row[:12] + [0.0, 0.0, 0.0] + [np.array(row[-1])]
            list_of_individuals.append(IndividualDTO(id=str(uuid.uuid4()),
                                                     params=self._int_check(np.array(row)),
                                                     dataset=self.dataset,
                                                     exp_id=self.exp_id))
        self.employed_bees = estimate_fitness(list_of_individuals)
        abc_fitness(self.employed_bees)
        self.fitness_evals += len(list_of_individuals)

    def _random_init_food(self,
                          resources_num=None):
        if resources_num is None:
            resources_num = self.food_resources_num
        list_of_individuals = []
        for _ in range(self.food_resources_num):
            params = self._init_random_params()
            list_of_individuals.append(IndividualDTO(id=str(uuid.uuid4()),
                                                     params=self._int_check(params),
                                                     dataset=self.dataset,
                                                     exp_id=self.exp_id))
        self.employed_bees = estimate_fitness(list_of_individuals)
        abc_fitness(self.employed_bees)
        self.fitness_evals += len(list_of_individuals)

    def _init_random_params(self):
        val_decor = np.random.uniform(low=self.low_decor, high=self.high_decor, size=1)[0]
        var_n = np.random.randint(low=self.low_n, high=self.high_n, size=5)
        var_sm = np.random.uniform(low=self.low_sm, high=self.high_sm, size=2)
        var_sp = np.random.uniform(low=self.low_sp, high=self.high_sp, size=4)
        var_b = np.random.randint(low=self.low_back, high=self.high_back, size=1)[0]
        val_decor_2 = np.random.uniform(low=0, high=1, size=1)[0]  # here
        params = [
            val_decor, var_n[0],
            var_sm[0], var_sm[1], var_n[1],
            var_sp[0], var_sp[1], var_n[2],
            var_sp[2], var_sp[3], var_n[3],
            var_b,
            0, 0, 0,  # to make compatability
            val_decor_2,
        ]
        params = [float(i) for i in params]
        return params

    def _explore_new_source(self, current_bee_idx):
        change_param = random.choice([i for i in range(12)] + [15])

        # getting the best solution
        solutions = [bee.fitness_value for bee in self.employed_bees]
        best_solution = np.argmax(solutions)
        best_solution_params = self.employed_bees[best_solution].params

        r = random.random()
        neighbour = int(r * self.food_resources_num)
        while neighbour == current_bee_idx:
            r = random.random()
            neighbour = int(r * self.food_resources_num)
        current_params = np.copy(self.employed_bees[current_bee_idx].params)
        # make r dependend on parameters
        r = random.uniform(-1, 1)
        #         current_params[change_param] = current_params[change_param] + r * (
        current_params[change_param] = best_solution_params[change_param] + r * (
                current_params[change_param] - self.employed_bees[neighbour].params[change_param])
        for i in range(len(current_params)):
            current_params[i] = self._check_param_limits(current_params, i)[i]
        return self._int_check(current_params)

    def _employed_bees_phase(self):
        new_employed_bees_solutions = []
        for bee_idx, bee in enumerate(self.employed_bees):
            current_params = self._explore_new_source(bee_idx)
            new_employed_bees_solutions.append(IndividualDTO(id=str(uuid.uuid4()),
                                                             params=current_params,
                                                             dataset=self.dataset,
                                                             exp_id=self.exp_id
                                                             ))
        new_employed_bees = estimate_fitness(new_employed_bees_solutions)
        abc_fitness(new_employed_bees)
        self.fitness_evals += len(new_employed_bees_solutions)
        for i in range(len(new_employed_bees)):
            if new_employed_bees[i].fitness_value > self.employed_bees[i].fitness_value:
                self.trials[i] = 0
                self.employed_bees[i] = copy.deepcopy(new_employed_bees[i])
            else:
                self.trials[i] += 1

    def _calculate_probabilities(self):
        fitness = []
        for bee in self.employed_bees:
            fitness.append(bee.fitness_value)
        fitness = np.array(fitness)
        self.probabilities = 0.9 * (fitness / np.sum(fitness)) + 0.1

    # TODO: check the count of onlooker bees
    def _onlooker_bees_phase(self):
        selected_sources = []
        new_onlooker_bees_solutions = []
        flag = True
        solution_indices = []
        counter = 0
        while flag:
            for bee_idx, prob in enumerate(self.probabilities):
                r = random.random()
                if r <= prob:
                    selected_sources.append(bee_idx)
                    current_params = self._explore_new_source(bee_idx)
                    new_onlooker_bees_solutions.append(IndividualDTO(id=str(uuid.uuid4()),
                                                                     params=current_params,
                                                                     dataset=self.dataset,
                                                                     exp_id=self.exp_id))
                    solution_indices.append(bee_idx)
                    self.fitness_evals += 1
                    counter += 1
                    if counter == self.food_resources_num:
                        flag = False
        new_onlooker_bees = estimate_fitness(new_onlooker_bees_solutions)
        abc_fitness(new_onlooker_bees)
        for idx, ix in enumerate(solution_indices):
            if new_onlooker_bees[idx].fitness_value > self.employed_bees[ix].fitness_value:
                self.employed_bees[ix] = new_onlooker_bees[idx]
                self.trials[ix] = 0
            else:
                self.trials[ix] += 1

    def _scout_bees_phase(self):
        new_scout_bees = []
        guys_to_remove = []
        for ix, trial in enumerate(self.trials):
            if trial >= self.max_num_trials:
                guys_to_remove.append(ix)
                new_scout_bees.append(IndividualDTO(id=str(uuid.uuid4()),
                                                    params=self._init_random_params(),
                                                    dataset=self.dataset,
                                                    exp_id=self.exp_id))
        if len(new_scout_bees) > 0:
            new_bees = estimate_fitness(new_scout_bees)
            abc_fitness(new_bees)
            self.fitness_evals += len(new_scout_bees)
            for ix, i in enumerate(guys_to_remove):
                self.employed_bees[i] = new_bees[ix]
                self.trials[i] = 0

    def _show_best_solution(self):
        fitness = []
        for bee in self.employed_bees:
            fitness.append(bee.fitness_value)
        fitness = np.array(fitness)
        best_fitness = np.max(fitness)
        if best_fitness > self.best_solution:
            self.best_solution = best_fitness
        logger.info(f'best fitness on current iteration: {best_fitness - 1}')
        logger.info(f'global best fitness: {self.best_solution - 1}')

    def run(self, iterations):
        self.init_resources_method()
        logger.info('Population is initialized')
        logger.info(f'Fitness counts: {self.fitness_evals}')

        random_search_res = np.max([bee.fitness_value for bee in self.employed_bees])

        for i in range(iterations):
            self._employed_bees_phase()
            logger.info('Employed bees phase is over')
            logger.info(f'Fitness counts: {self.fitness_evals}')

            if self.fitness_evals >= self.num_fitness_evaluations:
                self._show_best_solution()
                break

            self._calculate_probabilities()
            self._onlooker_bees_phase()
            logger.info('Onlooker bees phase is over')
            logger.info(f'Fitness counts: {self.fitness_evals}')

            if self.fitness_evals >= self.num_fitness_evaluations:
                self._show_best_solution()
                break

            self._scout_bees_phase()

            if self.fitness_evals >= self.num_fitness_evaluations:
                self._show_best_solution()
                break

            self._show_best_solution()
        return random_search_res

    def _check_param_limits(self, params, idx):
        try:
            if idx == 15:
                param_idx = 12
                if params[idx] < self.params_limits[param_idx][0]:
                    params[idx] = self.params_limits[param_idx][0]
                if params[idx] > self.params_limits[param_idx][1]:
                    params[idx] = self.params_limits[param_idx][1]
            else:
                if params[idx] < self.params_limits[idx][0]:
                    params[idx] = self.params_limits[idx][0]
                if params[idx] > self.params_limits[idx][1]:
                    params[idx] = self.params_limits[idx][1]
        except:
            pass
        return params

    def _int_check(self, params):
        res = list(params)
        for i in [1, 4, 7, 10, 11]:
            res[i] = float(np.round(res[i]))
        return res


if __name__ == "__main__":
    run_algorithm()
