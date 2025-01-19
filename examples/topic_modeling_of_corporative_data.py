import logging

import os
import uuid
from typing import Dict, Any, Optional

import pandas as pd
from sklearn.model_selection import train_test_split

from autotm.base import AutoTM

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.DEBUG, datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger()


def main():
    df = pd.read_csv('data/sample_corpora/clean_docs_v17_gost_only.csv')
    train_df, test_df = train_test_split(df, test_size=0.1)

    working_dir_path = f"./autotm_workdir_{uuid.uuid4()}"
    model_path = os.path.join(working_dir_path, "autotm_model")

    autotm = AutoTM(
        topic_count=50,
        texts_column_name='paragraph',
        preprocessing_params={
            "lang": "ru"
        },
        alg_name="ga",
        alg_params={
            "num_iterations": 10,
            "use_pipeline": True
        },
        individual_type="llm",
        working_dir_path=working_dir_path,
        exp_dataset_name="gost"
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
    main(conf_name="gost_example")
