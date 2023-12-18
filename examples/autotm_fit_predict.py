import logging
import os
import uuid

import pandas as pd
from sklearn.model_selection import train_test_split

from autotm.base import AutoTM


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()


def main():
    path_to_dataset = "data/sample_corpora/sample_dataset_lenta.csv"
    alg_name = "ga"

    df = pd.read_csv(path_to_dataset)
    train_df, test_df = train_test_split(df, test_size=0.1)

    working_dir_path = f"./autotm_workdir_{uuid.uuid4()}"
    model_path = os.path.join(working_dir_path, "autotm_model")

    autotm = AutoTM(
        topic_count=20,
        preprocessing_params={
            "lang": "ru",
            "min_tokens_count": 3
        },
        alg_name=alg_name,
        alg_params={
            "num_iterations": 2,
            "num_individuals": 10,
            "use_nelder_mead_in_mutation": False,
            "use_nelder_mead_in_crossover": False,
            "use_nelder_mead_in_selector": False,
            "train_option": "offline"
        },
        working_dir_path=working_dir_path,
        exp_dataset_name="lenta_ru"
    )
    mixtures = autotm.fit_predict(train_df)

    logger.info(f"Calculated train mixtures: {mixtures.shape}\n\n{mixtures.head(10).to_string()}")

    # saving the model
    autotm.save(model_path)

    # loading and checking if everything is fine with predicting
    autotm_loaded = AutoTM.load(model_path)
    mixtures = autotm_loaded.predict(test_df)

    logger.info(f"Calculated train mixtures: {mixtures.shape}\n\n{mixtures.head(10).to_string()}")


if __name__ == "__main__":
    main()
