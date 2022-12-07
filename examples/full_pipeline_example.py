# input example_data padas df with 'text' column
import os
import pandas as pd
import sys
from autotm.infer import (get_experiment_path, get_artifacts, get_most_probable_topics_from_theta,
                          get_top_words_from_topic_in_text)

from autotm.preprocessing.text_preprocessing import process_dataset
from autotm.preprocessing.dictionaries_preparation import prepare_all_artifacts
from autotm.algorithms_for_tuning.genetic_algorithm.genetic_algorithm import run_algorithm

PATH_TO_DATASET = '../data/sample_corpora/sample_dataset_lenta.csv'  # dataset with corpora to be processed
SAVE_PATH = '../data/processed_sample_corpora'  # place where all the artifacts will be stored

dataset = pd.read_csv(PATH_TO_DATASET)
col_to_process = 'text'
lang = 'ru'  # available languages: ru, en
min_tokens_num = 3  # the minimal amount of tokens after processing to save the result
exp_id = 1  # do not forget to change the experiment id for each run (will be fixed later)
num_iterations = 5
topic_count = 5

if __name__ == '__main__':
    print('Stage 1: Dataset preparation')
    process_dataset(PATH_TO_DATASET, col_to_process, SAVE_PATH,
                    lang, min_tokens_count=min_tokens_num)

    prepare_all_artifacts(SAVE_PATH)
    print('Stage 2: Tuning the topic model')

    # exp_id and dataset_name will be needed further to store results in mlflow
    best_result = run_algorithm(data_path=SAVE_PATH,
                                exp_id=exp_id,
                                dataset='test',
                                topic_count=topic_count,
                                log_file='./log_file_test.txt',
                                num_iterations=num_iterations
                                )

    # results of the run are stored in ./mlruns folder, experiment id is 'experiment_<exp_id>'


    # uncomment this after getting a topic model
    # print('Step 3: Looking at results and making inference')
    # # usage and inference
    #
    # TEST_RUN_NAME = 'fitness-test-50051a97-98d5-41b8-b4a7-e0e76566751a'  # run_name is shown in terminal after model run
    # MLFLOW_PATH = './mlruns/'
    # processed_dataset = pd.read_csv(
    #     os.path.join(SAVE_PATH, 'processed_dataset.csv'))  # here the processed dataset is needed
    # artifacts_path = get_experiment_path(exp_id, TEST_RUN_NAME)
    # if artifacts_path:
    #     phi_matrix, theta_matrix, topics = get_artifacts(artifacts_path)
    #     processed_dataset = get_most_probable_topics_from_theta(processed_dataset, theta_matrix)
    #     processed_dataset = get_top_words_from_topic_in_text(processed_dataset,
    #                                                          topics)  # SER_words_of_topic contains dictionaries with top tokens of the topic that were met in text
    #     print(processed_dataset)
