import time
import logging
import os
import uuid
import pandas as pd
import numpy as np
from datasets import load_dataset
from sklearn.model_selection import train_test_split
from autotm.base import AutoTM
from autotm.ontology.ontology_extractor import build_graph
import networkx as nx


df = pd.read_dataset('../data/sample_corpora/dataset_books_stroitelstvo.csv')

working_dir_path = 'autotm_artifacts'

autotm = AutoTM(
        topic_count=10,
        texts_column_name='text',
        preprocessing_params={
            "lang": "ru",
        },
        alg_params={
            "num_iterations": 10,
        },
        working_dir_path=working_dir_path
    )

mixtures = autotm.fit_predict(df)

# посмотрим на получаемые темы
autotm.print_topics()

# Составим словарь наименований тем! Обратите внимание, что есть вероятность получения других смесей тем при новом запуске
labels_dict = {'main0':'Кровля',
               'main1': 'Фундамент',
               'main2':'Техника строительства',
               'main3': 'Электросбережение',
               'main4':'Генплан',
               'main5': 'Участок',
               'main6': 'Лестница',
               'main7':'Внешняя отделка',
               'main8': 'Стены',
               'main9': 'Отделка (покраска)',
               }

res_dict, nodes = build_graph(autotm, topic_labels=labels_dict)

# визуализируем граф
g = nx.DiGraph()
g.add_nodes_from(nodes)
for k, v in res_dict.items():
    g.add_edges_from(([(k)]))
nx.draw(g, with_labels=True)