import logging
import os
import tempfile

import pandas as pd
from numpy.typing import ArrayLike

from autotm.base import AutoTM

logger = logging.getLogger(__name__)


def check_predictions(autotm: AutoTM, df: pd.DataFrame, mixtures: ArrayLike):
    n_samples, n_samples_mixture = df.shape[0], mixtures.shape[0]
    n_topics, n_topics_mixture = autotm.topics.shape[0], mixtures.shape[1]

    assert n_samples_mixture == n_samples
    assert n_topics_mixture == n_topics


def test_fit_predict():
    # dataset with corpora to be processed
    path_to_dataset = "../../data/sample_corpora/sample_dataset_lenta.csv"
    alg_name = "ga"

    df = pd.read_csv(path_to_dataset)

    with tempfile.TemporaryDirectory(prefix="fp_tmp_working_dir_") as tmp_working_dir:
        model_path = os.path.join(tmp_working_dir, "autotm_model")

        autotm = AutoTM(
            preprocessing_params={
                "lang": "ru",
                "min_tokens_count": 3
            },
            alg_name=alg_name,
            alg_params={
                "num_iterations": 10,
                "use_nelder_mead_in_mutation": False,
                "use_nelder_mead_in_crossover": False,
                "use_nelder_mead_in_selector": False
            },
            artm_train_options={"mode": "offline"},
            working_dir_path=tmp_working_dir
        )
        mixtures = autotm.fit_predict(df)
        check_predictions(autotm, df, mixtures)

        # saving the model
        autotm.save(model_path)

        # loading and checking if everything is fine with predicting
        autotm_loaded = AutoTM.load(model_path)
        mixtures = autotm_loaded.predict(df)
        check_predictions(autotm, df, mixtures)
