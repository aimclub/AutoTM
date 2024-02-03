import datetime
import os
import re
import subprocess
import sys
import time
import warnings

from pydantic import PydanticDeprecatedSince20

from autotm.algorithms_for_tuning.genetic_algorithm.genetic_algorithm import get_best_individual

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=PydanticDeprecatedSince20)

SAVE_PATH = "/Users/Maksim.Zuev/Downloads/AutoTM/"

datasets = ["20newsgroups_sample", "amazon_food_sample", "banners_sample", "hotel-reviews_sample", "lenta_ru_sample"]
num_iterations = 500
num_fitness_evaluations = 150
num_individuals = 11
topic_count = 10
use_nelder_mead_in_mutation = False
use_nelder_mead_in_crossover = False
use_nelder_mead_in_selector = False
train_option = "offline"


def run_single_experiment(dataset_name, date, use_pipeline: bool):
    exp_id = int(time.time())
    _, stats = get_best_individual(
        data_path=f"{SAVE_PATH}/{dataset_name}",
        dataset=dataset_name,
        exp_id=exp_id,
        topic_count=topic_count,
        log_file=f"./log-{date}_autotm.txt",
        num_iterations=num_iterations,
        num_individuals=num_individuals,
        num_fitness_evaluations=num_fitness_evaluations,
        use_pipeline=use_pipeline,
        use_nelder_mead_in_mutation=use_nelder_mead_in_mutation,
        use_nelder_mead_in_crossover=use_nelder_mead_in_crossover,
        use_nelder_mead_in_selector=use_nelder_mead_in_selector,
        train_option=train_option,
        quiet_log=True
    )
    return stats


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
                sys.stdout = save_stdout
                sys.stderr = save_stderr
    except Exception as e:
        print("Error: ", e)
    finally:
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

    date = datetime.datetime.now().strftime('%y%m%d-%H%M%S')
    file_name = f"log-{date}.txt"
    with open(file_name, 'a') as logfile:
        for dataset_name in datasets:
            for use_pipeline in [False, True]:
                for _ in range(10):
                    start_time = time.time()
                    used_fitness, best_fitness = suppress_stdout(lambda: run_single_experiment(dataset_name, date, use_pipeline))
                    elapsed_time = time.time() - start_time
                    elapsed_minutes, elapsed_seconds = divmod(int(elapsed_time), 60)
                    stats = ", ".join(f"({i}, {fi})" for i, fi in zip(used_fitness, best_fitness))
                    print(f"Time: {elapsed_minutes} min {elapsed_seconds} sec, use_pipeline: {use_pipeline}, dataset: {dataset_name}, (i, f(i)): {stats}")
                    print(f'{git_hash},{dataset_name},{use_pipeline},{used_fitness},{best_fitness},{num_fitness_evaluations},{num_iterations},{num_individuals},{topic_count},{train_option}',
                          file=logfile)


if __name__ == "__main__":
    main()
