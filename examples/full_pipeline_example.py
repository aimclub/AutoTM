# input example_data padas df with 'text' column
import os
import pandas as pd

from src.autotm.preprocessing.text_preprocessing import process_dataset

PATH_TO_DATASET = '../data/sample_corpora/sample_dataset_lenta.csv'
SAVE_PATH = '../data/processed_sample_corpora' # place where all the artifacts will be stored

dataset = pd.read_csv(PATH_TO_DATASET)
lang = 'ru'
col_to_process = 'text'
min_tokens_num = 3

print('Stage 1: Dataset preparation')
process_dataset(PATH_TO_DATASET, col_to_process, SAVE_PATH, lang)
