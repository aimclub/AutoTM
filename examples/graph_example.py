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


df = load_dataset('zloelias/lenta-ru')  # https://huggingface.co/datasets
text_sample = df['train']['text'][:5000]
df = pd.DataFrame({'text': text_sample})

working_dir_path = 'autotm_artifacts'

autotm = AutoTM(
        topic_count=20,
        texts_column_name='text',
        preprocessing_params={
            "lang": "ru",
        },
        # alg_name='ga',
        alg_params={
            "num_iterations": 10,
        },
        working_dir_path=working_dir_path
    )

mixtures = autotm.fit_predict(df)

labels_dict = {'main0':'Устройства',
               'main1': 'Спорт',
               'main2':'Экономика',
               'main3': 'Чемпионат',
               'main4':'Исследование',
               'main5': 'Награждение',
               'main6': 'Суд',
               'main7':'Общее',
               'main8': 'Авиаперелеты',
               'main9': 'Музеи',
               'main10': 'Правительство',
               'main11': 'Интернет',
               'main12': 'Искусство',
               'main13': 'Война',
               'main14': 'Нефть',
               'main15': 'Космос',
               'main16': 'Соревнования',
               'main17': 'Биржа',
               'main18': 'Финансы',
               'main19': 'Концерт'
               }

res_dict, nodes = build_graph(autotm, topic_labels=labels_dict)

# visualizing the graph
g = nx.DiGraph()
g.add_nodes_from(nodes)
for k, v in res_dict.items():
    g.add_edges_from(([(k)]))
nx.draw(g, with_labels=True)