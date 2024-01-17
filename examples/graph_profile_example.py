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
from autotm.clustering import cluster_phi
import networkx as nx


df = load_dataset('zloelias/lenta-ru')  # https://huggingface.co/datasets
text_sample = df['train']['text']
df = pd.DataFrame({'text': text_sample})

working_dir_path = 'autotm_artifacts'

# Инициализируем модель и укажем наименование
# суррогатной модели для ускорения вычислений
autotm = AutoTM(
        topic_count=20,
        texts_column_name='text',
        preprocessing_params={
            "lang": "ru",
        },
        # alg_name='ga',
        alg_params={
            "num_iterations": 15,
            "surrogate_alg_name": "GPR"
        },
        working_dir_path=working_dir_path
    )

mixtures = autotm.fit_predict(df)

# извлекаем матрицу phi - распределение слов по темам
phi_df = autotm._model.get_phi()

# проведем кластеризацию полученных данных
# из которой можно увидеть, что достаточно хорошо выделяются кластера
cluster_phi(phi_df, n_clusters=10, plot_img=True)

# осмотрим, какие документы есть в кластерах
mixtures['labels'] = y_kmeans.labels_
res_df = df.join(mixtures)
print(res_df[res_df['labels'] == 8].text.tolist()[:3])

# займемся построением графа, как мы уже знаем
# для лучшей интерпретируемости лучше сделать
# словарь названий тем
autotm.print_topics()
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

# посмотрим на результат обработки - словарь со связями
print(res_dict)

# Визуализируем получаемый граф
g = nx.DiGraph()
g.add_nodes_from(nodes)
for k, v in res_dict.items():
    g.add_edges_from(([(k)]))
nx.draw(g, with_labels=True)

# Раскроем одну из нод
mixtures_subset = mixtures.sort_values('main0', ascending=False).head(1000)

subset_df = df.join(mixtures_subset)

autotm_subs = AutoTM(
        topic_count=6,
        texts_column_name='text',
        preprocessing_params={
            "lang": "ru",
        },
        alg_params={
            "num_iterations": 10,
            "surrogate_alg_name": "GPR"
        },
        working_dir_path=working_dir_path
    )

autotm_subs.print_topics()
subs_labels = {
    'main0': 'Технологии',
    'main1': 'Разработка',
    'main2': 'Игры',
    'main3': 'Компьютеры',
    'main4': 'Мобильные',
    'main5': 'Переферия'
}

autotm_subs.fit_predict(subset_df)
res_dict_subs, nodes_subs = build_graph(autotm_subs, topic_labels=subs_labels)

g = nx.DiGraph()
g.add_nodes_from(nodes_subs)
for k, v in res_dict_subs.items():
    g.add_edges_from(([(k)]))
nx.draw(g, with_labels=True)