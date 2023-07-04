import logging
import os
import tempfile

import pandas as pd
from numpy.typing import ArrayLike
from sklearn.model_selection import train_test_split

from autotm.base import AutoTM

logger = logging.getLogger(__name__)


def check_predictions(autotm: AutoTM, df: pd.DataFrame, mixtures: ArrayLike):
    n_samples, n_samples_mixture = df.shape[0], mixtures.shape[0]
    n_topics, n_topics_mixture = len(autotm.topics), mixtures.shape[1]

    assert n_samples_mixture == n_samples
    assert n_topics_mixture == n_topics
    assert (~mixtures.isna()).all().all()
    assert (~mixtures.isnull()).all().all()


def test_fit_predict():
    # dataset with corpora to be processed
    path_to_dataset = "data/sample_corpora/sample_dataset_lenta.csv"
    alg_name = "ga"

    df = pd.read_csv(path_to_dataset)
    train_df, test_df = train_test_split(df, test_size=0.1)

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
