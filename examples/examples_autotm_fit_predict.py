import logging

import os
import uuid
from typing import Dict, Any

import pandas as pd
from sklearn.model_selection import train_test_split

from autotm.base import AutoTM

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.DEBUG, datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger()

CONFIGURATIONS = {
    "base": {
        "alg_name": "ga",
        "num_iterations": 2,
        "num_individuals": 2,
        "use_pipeline": True
    },
    "static_chromosome": {
        "alg_name": "ga",
        "num_iterations": 2,
        "num_individuals": 2,
        "use_pipeline": False
    },
    "surrogate": {
        "alg_name": "ga",
        "num_iterations": 2,
        "num_individuals": 2,
        "use_pipeline": True,
        "surrogate_name": "random-forest-regressor"
    },
    "llm": {
        "alg_name": "ga",
        "num_iterations": 2,
        "num_individuals": 2,
        "use_pipeline": True,
        "individual_type": "llm"
    },
    "bayes": {
        "alg_name": "bayes",
        "num_evaluations": 5,
    }
}


def run(alg_name: str, alg_params: Dict[str, Any]):
    path_to_dataset = "data/sample_corpora/sample_dataset_lenta.csv"

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
        alg_params=alg_params,
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


def main(conf_name: str = "base"):
    if conf_name not in CONFIGURATIONS:
        raise ValueError(
            f"Unknown configuration {conf_name}. Available configurations: {sorted(CONFIGURATIONS.keys())}"
        )

    conf = CONFIGURATIONS[conf_name]
    alg_name = conf['alg_name']
    del conf['alg_name']

    run(alg_name=alg_name, alg_params=conf)


if __name__ == "__main__":
    main()