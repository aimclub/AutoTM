# input example_data padas df with 'text' column
import os
import pandas as pd
import sys

from autotm.preprocessing.text_preprocessing import process_dataset
from autotm.preprocessing.dictionaries_preparation import prepare_all_artifacts
from autotm.algorithms_for_tuning.genetic_algorithm.genetic_algorithm import run_algorithm

PATH_TO_DATASET = '../data/sample_corpora/sample_dataset_lenta.csv'
SAVE_PATH = '../data/processed_sample_corpora'  # place where all the artifacts will be stored

dataset = pd.read_csv(PATH_TO_DATASET) # dataset with corpora to be processed
col_to_process = 'text'
lang = 'ru'  # available languages: ru, en
min_tokens_num = 3 # the minimal amount of tokens after processing to save the result

if __name__ == '__main__':
    print('Stage 1: Dataset preparation')
    process_dataset(PATH_TO_DATASET, col_to_process, SAVE_PATH,
                    lang, min_tokens_count=min_tokens_num)

    prepare_all_artifacts(SAVE_PATH)

    print('Stage 2: Tuning the topic model')

    # exp_id and dataset_name will be needed further to store results in mlflow
    topics = run_algorithm(data_path=SAVE_PATH,
                           exp_id=1,
                           dataset='test',
                           topic_count=3,
                           log_file='./log_file_test.txt',
                           num_iterations=2
                           )
