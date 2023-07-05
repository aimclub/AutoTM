# input example_data padas df with 'text' column
import time

import pandas as pd

from autotm.algorithms_for_tuning.genetic_algorithm.genetic_algorithm import (
    run_algorithm,
)
from autotm.preprocessing.dictionaries_preparation import prepare_all_artifacts
from autotm.preprocessing.text_preprocessing import process_dataset

PATH_TO_DATASET = "../data/sample_corpora/sample_dataset_lenta.csv"  # dataset with corpora to be processed
SAVE_PATH = (
    "../data/processed_sample_corpora"  # place where all the artifacts will be stored
)

dataset = pd.read_csv(PATH_TO_DATASET)
col_to_process = "text"
dataset_name = "lenta_sample"
lang = "ru"  # available languages: ru, en
min_tokens_num = 3  # the minimal amount of tokens after processing to save the result
num_iterations = 2
topic_count = 10
exp_id = int(time.time())
print(exp_id)

use_nelder_mead_in_mutation = False
use_nelder_mead_in_crossover = False
use_nelder_mead_in_selector = False
train_option = "offline"

if __name__ == "__main__":
    print("Stage 1: Dataset preparation")
    process_dataset(
        PATH_TO_DATASET,
        col_to_process,
        SAVE_PATH,
        lang,
        min_tokens_count=min_tokens_num,
    )
    prepare_all_artifacts(SAVE_PATH)
    print("Stage 2: Tuning the topic model")

    # exp_id and dataset_name will be needed further to store results in mlflow
    best_result = run_algorithm(
        data_path=SAVE_PATH,
        dataset=dataset_name,
        exp_id=exp_id,
        topic_count=topic_count,
        log_file="./log_file_test.txt",
        num_iterations=num_iterations,
        use_nelder_mead_in_mutation=use_nelder_mead_in_mutation,
        use_nelder_mead_in_crossover=use_nelder_mead_in_crossover,
        use_nelder_mead_in_selector=use_nelder_mead_in_selector,
        train_option=train_option,
    )
