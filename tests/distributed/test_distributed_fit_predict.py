import os
import tempfile
from typing import Dict

import pandas as pd

from docker.models.containers import Container
from sklearn.model_selection import train_test_split

from autotm.base import AutoTM
from ..utils import check_predictions


def test_distributed_fit_predict(pytestconfig, fitness_computing_settings: Dict[str, str]):
    # todo: may be not applied correctly due to too late creating
    # settings for distributed run
    os.environ['AUTOTM_COMPONENT'] = 'head'
    os.environ['AUTOTM_EXEC_MODE'] = 'cluster'
    os.environ['CELERY_BROKER_URL'] = 'amqp://guest:guest@localhost:5672'
    os.environ['CELERY_RESULT_BACKEND'] = 'redis://localhost:6379/1'

    # dataset with corpora to be processed
    path_to_dataset = os.path.join(pytestconfig.rootpath, "data/sample_corpora/sample_dataset_lenta.csv")
    alg_name = "ga"

    df = pd.read_csv(path_to_dataset)
    train_df, test_df = train_test_split(df, test_size=0.1)

    # todo: ensure somehow that the results have been calculated on remote fitness-worker
    with tempfile.TemporaryDirectory(prefix="fp_tmp_working_dir_") as tmp_working_dir:
        model_path = os.path.join(tmp_working_dir, "autotm_model")

        autotm = AutoTM(
            preprocessing_params={
                "lang": "ru",
                "min_tokens_count": 3
            },
            alg_name=alg_name,
            alg_params={
                "num_iterations": 2,
                "num_individuals": 4,
                "use_nelder_mead_in_mutation": False,
                "use_nelder_mead_in_crossover": False,
                "use_nelder_mead_in_selector": False,
                "train_option": "offline"
            },
            working_dir_path=tmp_working_dir
        )
        mixtures = autotm.fit_predict(train_df)
        check_predictions(autotm, train_df, mixtures)

        # saving the model
        autotm.save(model_path)

        # loading and checking if everything is fine with predicting
        autotm_loaded = AutoTM.load(model_path)
        mixtures = autotm_loaded.predict(test_df)
        check_predictions(autotm, test_df, mixtures)
