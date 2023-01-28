from autotm.algorithms_for_tuning.nelder_mead_optimization.nelder_mead import NelderMeadOptimization
import time
import pandas as pd
import os

# 2

fnames = ['20newsgroups_sample', 'amazon_food_sample', 'banners_sample', 'hotel-reviews_sample', 'lenta_ru_sample']
data_lang = ['en', 'en', 'ru', 'en', 'ru']
dataset_id = 1

DATA_PATH = os.path.join('/ess_data/GOTM/datasets_TM_scoring', fnames[dataset_id])

PATH_TO_DATASET = os.path.join(DATA_PATH, 'dataset_processed.csv')  # dataset with corpora to be processed
SAVE_PATH = DATA_PATH  # place where all the artifacts will be stored

dataset = pd.read_csv(PATH_TO_DATASET)
dataset_name = fnames[dataset_id] + 'sample_with_nm'
lang = data_lang[dataset_id]  # available languages: ru, en
min_tokens_num = 3  # the minimal amount of tokens after processing to save the result
num_iterations = 20
topic_count = 10
exp_id = int(time.time())
print(exp_id)
train_option = 'offline'

if __name__ == '__main__':
    nelder_opt = NelderMeadOptimization(data_path=SAVE_PATH,
                                        dataset=dataset_name,
                                        exp_id=exp_id,
                                        topic_count=topic_count,
                                        train_option=train_option)

    res = nelder_opt.run_algorithm(num_iterations=num_iterations)
    print(res)
    print(-res.fun)
