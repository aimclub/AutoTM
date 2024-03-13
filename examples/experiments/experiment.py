import datetime
import os
import re
import subprocess
import sys
import time
import warnings

import numpy as np
from autotm.algorithms_for_tuning.genetic_algorithm.statistics_collector import StatisticsCollector
from pydantic import PydanticDeprecatedSince20

from autotm.algorithms_for_tuning.genetic_algorithm.genetic_algorithm import get_best_individual
from autotm.algorithms_for_tuning.individuals import Individual

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=PydanticDeprecatedSince20)

SAVE_PATH = "/Users/Maksim.Zuev/PycharmProjects/AutoTMResources/datasets"

datasets = [
    "hotel-reviews_sample",
    "lenta_ru_sample"
    "amazon_food_sample",
    "20newsgroups_sample",
    "banners_sample",
]
num_iterations = 500
num_fitness_evaluations = 150
num_individuals = 11
topic_count = 10
use_nelder_mead_in_mutation = False
use_nelder_mead_in_crossover = False
use_nelder_mead_in_selector = False
train_option = "offline"


def replace_with_max(array):
    array = np.array(array)
    array = np.maximum.accumulate(array)
    return array.tolist()


def transform_to_invocations(iterations, fitness):
    invocations = list(range(iterations[0], iterations[-1] + 1))
    new_fitness = [0] * len(invocations)
    min_inv = iterations[0]
    for i in range(len(iterations) - 1):
        inv = iterations[i]
        next_inv = iterations[i + 1]
        for j in range(inv - min_inv, next_inv - min_inv):
            new_fitness[j] = fitness[i]
    new_fitness[-1] = fitness[-1]
    cut_length = min(len(invocations), num_fitness_evaluations - min_inv + 1)
    return invocations[:cut_length], new_fitness[:cut_length]


class FileStatisticsCollector(StatisticsCollector):
    def __init__(self, base_dir, dataset: str, use_pipeline: bool):
        self.dataset = dataset
        self.use_pipeline = use_pipeline
        date = datetime.datetime.now().strftime('%y%m%d-%H%M%S')
        mode = "pipeline" if use_pipeline else "fixed"
        experiment_name = f"{date}_{dataset}_{mode}"
        self.logs_path = os.path.join(base_dir, f"logs/{experiment_name}_autotm.log")
        os.makedirs(os.path.join(base_dir, "logs"), exist_ok=True)
        self.progress_path = os.path.join(base_dir, f"statistics/{experiment_name}_progress.txt")
        self.parameters_path = os.path.join(base_dir, f"statistics/{experiment_name}_parameters.txt")
        os.makedirs(os.path.join(base_dir, "statistics"), exist_ok=True)
        self.parameters_file = open(self.parameters_path, 'w')

        self.best_fitness = -1
        self.evaluations = []
        self.fitness = []

    def log_iteration(self, evaluations: int, best_fitness: float):
        self.evaluations.append(evaluations)
        self.fitness.append(best_fitness)
        self.best_fitness = max(best_fitness, self.best_fitness)

    def log_individual(self, individual: Individual):
        json = individual.dto.model_dump_json()
        print(json, file=self.parameters_file)

    def finalize(self):
        self.parameters_file.close()
        invocations, fitness = transform_to_invocations(self.evaluations, replace_with_max(self.fitness))
        with open(self.progress_path, 'w') as file:
            for i, f in zip(invocations, fitness):
                print(f"{self.dataset},{self.use_pipeline},{i},{f}", file=file)


def run_single_experiment(base_dir, dataset_name, use_pipeline: bool):
    collector = FileStatisticsCollector(base_dir, dataset_name, use_pipeline)
    get_best_individual(
        data_path=f"{SAVE_PATH}/{dataset_name}",
        dataset=dataset_name,
        exp_id=(int(time.time())),
        topic_count=topic_count,
        log_file=collector.logs_path,
        num_iterations=num_iterations,
        num_individuals=num_individuals,
        num_fitness_evaluations=num_fitness_evaluations,
        use_pipeline=use_pipeline,
        use_nelder_mead_in_mutation=use_nelder_mead_in_mutation,
        use_nelder_mead_in_crossover=use_nelder_mead_in_crossover,
        use_nelder_mead_in_selector=use_nelder_mead_in_selector,
        train_option=train_option,
        quiet_log=True,
        statistics_collector=collector,
        surrogate_name="random-forest-regressor",
    )
    collector.finalize()
    return collector


def assert_no_uncommitted_changes():
    uncommitted_changes = subprocess.check_output(["git", "status", "--porcelain"]).decode('utf-8').strip()
    assert not uncommitted_changes, f'There are uncommitted changes, please commit them first. {uncommitted_changes}'


def get_git_hash():
    return subprocess.check_output(["git", "rev-parse", "HEAD"]).strip()


def suppress_stdout(action):
    save_stdout = sys.stdout
    save_stderr = sys.stderr
    stdout_file_name = "stdout.txt"
    stderr_file_name = "stderr.txt"
    try:
        with open(stdout_file_name, 'a') as out:
            with open(stderr_file_name, 'a') as err:
                sys.stdout = out
                sys.stderr = err
                result = action()
    finally:
        sys.stdout = save_stdout
        sys.stderr = save_stderr
        if os.path.exists(stdout_file_name) and os.path.getsize(stdout_file_name) == 0:
            os.remove(stdout_file_name)
        if os.path.exists(stderr_file_name):
            with open(stderr_file_name, 'r') as file:
                content = file.read()
                if re.match(r'((^\d+it \[\d+:\d+, \?(\d+(\.\d+)?)?it/s\])?\n$)*', content):
                    os.remove(stderr_file_name)
    return result


def main():
    assert_no_uncommitted_changes()
    git_hash = get_git_hash()
    print(f'Git hash: {git_hash}')

    for dataset_name in datasets:
        for use_pipeline in [False, True]:
            for _ in range(10):
                start_time = time.time()
                collector = suppress_stdout(lambda: run_single_experiment(os.path.curdir, dataset_name, use_pipeline))
                elapsed_time = time.time() - start_time
                minutes, seconds = divmod(int(elapsed_time), 60)
                best_fitness = collector.best_fitness
                print(f"Time: {minutes} min {seconds} sec, use_pipeline: {use_pipeline}, dataset: {dataset_name}, "
                      f"fitness: {best_fitness}")


if __name__ == "__main__":
    main()
