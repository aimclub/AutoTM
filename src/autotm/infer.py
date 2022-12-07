import os
import yaml
import json

import pandas as pd

PATH_TO_RUN_NAME = 'tags/mlflow.runName'
PATH_TO_EXPERIMENT_ID = ''


def get_experiment_path(mlflow_path, exp_id, run_name):
    for folder in os.listdir(mlflow_path):
        if folder != '.trash':
            subfolders = os.listdir(os.path.join(mlflow_path, folder))
            if len(subfolders) > 1:
                with open(os.path.join(mlflow_path, folder, 'meta.yaml'), 'r') as f:
                    data = yaml.load(f)
                if data['name'] == f'experiment_{exp_id}':
                    for subfolder in subfolders:
                        with open(os.path.join(mlflow_path, folder, subfolder, PATH_TO_RUN_NAME)) as f:
                            f_run_name = f.read()
                            if f_run_name.strip() == run_name:
                                return os.path.join(mlflow_path, folder, subfolder, 'artifacts')
    return None


def get_artifacts(artifacts_path):
    phi_folders = os.listdir(os.path.join(artifacts_path, 'phi.csv'))
    theta_folders = os.listdir(os.path.join(artifacts_path, 'theta.csv'))
    phi_matrix = pd.read_csv(os.path.join(artifacts_path, 'phi.csv', phi_folders[0]))
    theta_matrix = pd.read_csv(os.path.join(artifacts_path, 'theta.csv', theta_folders[0]))
    with open(os.path.join(artifacts_path, 'topics.json')) as f:
        topics = json.load(f)
    return phi_matrix, theta_matrix, topics


# def _get_phi_dict_format(df):
#

def get_most_probable_topics_from_theta(df, theta_df):

    theta_df
    pass


def get_most_probable_words_from_phi(df, phi_df):
    pass
