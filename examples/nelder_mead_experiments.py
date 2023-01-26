from autotm.algorithms_for_tuning.nelder_mead_optimization.nelder_mead import NelderMeadOptimization
import time
import pandas as pd
import os

DATA_PATH = '../data/experiment_datasets/20newsgroups_sample'

PATH_TO_DATASET = os.path.join(DATA_PATH, 'dataset_processed.csv')  # dataset with corpora to be processed
SAVE_PATH = DATA_PATH # place where all the artifacts will be stored


dataset = pd.read_csv(PATH_TO_DATASET)
col_to_process = 'text'
dataset_name = 'sample_lenta'
lang = 'en'  # available languages: ru, en
min_tokens_num = 3  # the minimal amount of tokens after processing to save the result
num_iterations = 2
topic_count = 10
exp_id = int(time.time())
print(exp_id)

if __name__ == '__main__':
    nelder_opt = NelderMeadOptimization(data_path=SAVE_PATH,
                                dataset=dataset_name,
                                exp_id=exp_id,
                                topic_count=topic_count)

    nelder_opt.run_algorithm(num_individuals=10,
                             num_iterations=num_iterations)