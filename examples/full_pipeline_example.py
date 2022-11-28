# input example_data padas df with 'text' column
import os
import pandas as pd

from src.autotm.preprocessing.text_preprocessing import process_dataset
from src.autotm.preprocessing.dictionaries_preparation import prepare_all_artifacts
from src.autotm.algorithms_for_tuning.genetic_algorithm.genetic_algorithm import run_algorithm

PATH_TO_DATASET = '../data/sample_corpora/sample_dataset_lenta.csv'
SAVE_PATH = '../data/processed_sample_corpora' # place where all the artifacts will be stored

dataset = pd.read_csv(PATH_TO_DATASET)
lang = 'ru'
col_to_process = 'text'
min_tokens_num = 3

if __name__ == '__main__':
    print('Stage 1: Dataset preparation')
    process_dataset(PATH_TO_DATASET, col_to_process, SAVE_PATH,
                    lang, min_tokens_count=min_tokens_num)
    prepare_all_artifacts(SAVE_PATH)

    print('Stage 2: Tuning the topic model')
    # run_algorithm
    # run_algorithm()