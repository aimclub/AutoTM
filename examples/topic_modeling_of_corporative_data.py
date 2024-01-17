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


df = pd.read_dataset('../data/sample_corpora/clean_docs_v17_gost_only.csv')

working_dir_path = 'autotm_artifacts'

autotm = AutoTM(
        topic_count=50,
        texts_column_name='paragraph',
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