import logging
import os
import pickle
import shutil
import tempfile
import uuid
from typing import Union, Optional, Any, Dict, List

import artm
import pandas as pd
from sklearn.base import BaseEstimator

import warnings

from autotm.preprocessing import PREPOCESSED_DATASET_FILENAME

# TODO: Suppressing of DeprecationWarnings that are raise if we are running with __main__, need to research further
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    from autotm.algorithms_for_tuning.bayesian_optimization import bayes_opt

from autotm.algorithms_for_tuning.genetic_algorithm import genetic_algorithm
from autotm.fitness.tm import extract_topics, print_topics
from autotm.infer import TopicsExtractor
from autotm.preprocessing.dictionaries_preparation import prepare_all_artifacts
from autotm.preprocessing.text_preprocessing import process_dataset, PROCESSED_TEXT_COLUMN

logger = logging.getLogger(__name__)


class AutoTM(BaseEstimator):
    _ARTM_MODEL_FILENAME = "artm_model"
    _AUTOTM_DATA_FILENAME = "autotm_data"
    _SUPPORTED_ALGS = ["ga", "bayes"]

    @classmethod
    def load(cls, path: str) -> 'AutoTM':
        """
        Loads AutoTM instance from a path on local filesystem.
        :param path: a local filesystem path to load an AutoTM instance from.
        """
        assert os.path.exists(path), f"Path doesn't exist: {path}"

        artm_model_path = os.path.join(path, cls._ARTM_MODEL_FILENAME)
        autotm_data_path = os.path.join(path, cls._AUTOTM_DATA_FILENAME)

        if not (os.path.exists(artm_model_path) and os.path.exists(autotm_data_path)):
            raise FileNotFoundError(f"One or two of the follwing paths don't exist: "
                                    f"{artm_model_path}, {autotm_data_path}")

        model = artm.load_artm_model(artm_model_path)

        with open(autotm_data_path, "rb") as f:
            params = pickle.load(f)

        autotm = AutoTM(**params)
        autotm._model = model

        return autotm

    def __init__(self,
                 topic_count: int = 10,
                 preprocessing_params: Optional[Dict[str, Any]] = None,
                 alg_name: str = "ga",
                 alg_params: Optional[Dict[str, Any]] = None,
                 surrogate_alg_name: Optional[str] = None,
                 surrogate_alg_params: Optional[Dict[str, Any]] = None,
                 artm_train_options: Optional[Dict[str, Any]] = None,
                 working_dir_path: Optional[str] = None,
                 texts_column_name: str = "text",
                 log_file_path: Optional[str] = None,
                 exp_id: Optional[str] = None,
                 exp_tag: Optional[str] = None,
                 exp_dataset_name: Optional[str] = None
                 ):
        """
        :param topic_count: Count of topics to fit ARTM model with
        :param preprocessing_params: A dict with params for the preprocessor
        :param alg_name: An algorithm to use for hyper parameters tuning of ARTM model (available: ga, bayes)
        :param alg_params: A dict with the algorithm specific parameters. Depends on alg_name.
            If not specified default parameters will be used.
        :param surrogate_alg_name: An algorithm to use for surrogate training during hyperparameter optimization
            to reduce number of fitness estimations with real ARTM model fitting.
            If not specifed, no surrogates will be used for hyper parameters search.
        :param surrogate_alg_params: A dict with the surrogate algorithm specific parameters.
            Depends on surrogate_alg_name. Should not be specified if surrogate_alg_name is not defined.
        :param artm_train_options: A dict with additional training options for underlying BigARTM implementation.
        :param working_dir_path: A directory where a nested temporary folder is created
            to store intermediate BigARTM files and other supplementary files.
            By default, working directory the current process is running with.
        :param texts_column_name: A name of the column in Pandas DataFrame to read texts of the dataset,
            if the dataset is represented as a 'pd.DataFrame'.
        :param log_file_path: A file path to log file for an optimization process.
        :param exp_id: An experiment id for an experiment versioning system (for example, Mlflow)
            to use for reporting final and intermediate results and metrics. Only Mlflow is currently supported.
        :param exp_tag: An experiment tag to log into Mlflow for later search purposes.
        :param exp_dataset_name: A dataset name to log into Mlflow for later search purposes.
        """
        self.topic_count = topic_count
        self.preprocessing_params = preprocessing_params
        self.alg_name = alg_name
        self.alg_params = alg_params
        self.surrogate_alg_name = surrogate_alg_name
        self.surrogate_alg_params = surrogate_alg_params
        self.artm_train_options = artm_train_options
        self.working_dir_path = working_dir_path
        self.texts_column_name = texts_column_name
        self.log_file_path = log_file_path
        self.exp_id = exp_id
        self.exp_tag = exp_tag
        self.exp_dataset_name = exp_dataset_name
        self._model: Optional[artm.ARTM] = None

    def fit(self, dataset: Union[pd.DataFrame, pd.Series], processed_dataset_path: Optional[str] = None) -> 'AutoTM':
        """
        Preprocess texts in the datasets, looks for the best hyperparameters for ARTM model and fits the model
        with these parameters. The instance will contain topics with the most probable words belonging to them.

        Parameters
        ----------
        dataset : DataFrame, Series containing texts of the corpus of size 'n_samples'.
            If dataset is DataFrame, the column containing texts will be identified
            from value of 'self.texts_column_name'.

        Returns
        -------
        self : object
            Fitted Estimator.
            :param processed_dataset_path: optional path where to write intermediate processed dataset

        """
        self._check_if_already_fitted(fit_is_ok=False)

        processed_dataset_path = processed_dataset_path or os.path.join(self.working_dir_path, f"{uuid.uuid4()}")

        logger.info(f"Stage 0: Create working dir {self.working_dir_path} if not exists")

        os.makedirs(self.working_dir_path, exist_ok=True)

        logger.info("Stage 1: Dataset preparation")
        # TODO: convert Series to DataFrame
        process_dataset(
            dataset,
            self.texts_column_name,
            processed_dataset_path,
            **self.preprocessing_params
        )
        prepare_all_artifacts(processed_dataset_path)
        logger.info("Stage 2: Tuning the topic model")

        if self.alg_name not in self._SUPPORTED_ALGS:
            raise ValueError(f"Alg {self.alg_name} is not supported. "
                             f"Only the following algorithms are supported: {self._SUPPORTED_ALGS}")

        if self.alg_name == "ga":
            # TODO: add checking of surrogate alg names
            # TODO: make mlflow arguments optional
            # exp_id and dataset_name will be needed further to store results in mlflow
            best_topic_model = genetic_algorithm.run_algorithm(
                data_path=processed_dataset_path,
                dataset=self.exp_dataset_name or "__noname__",
                exp_id=self.exp_id or "0",
                topic_count=self.topic_count,
                log_file=self.log_file_path,
                **self.alg_params
            )
        else:
            # TODO: refactor this function
            best_topic_model = bayes_opt.run_algorithm(
                dataset=processed_dataset_path,
                log_file=self.log_file_path,
                exp_id=self.exp_id or "0",
                **self.alg_params
            )

        self._model = best_topic_model.model

        return self

    def predict(self, dataset: Union[pd.DataFrame, pd.Series]) -> pd.DataFrame:
        """
        Looks for the best hyperparameters for ARTM model, fits the model with these parameters
        and predict topics mixtures for individual documents in the incoming corpus.

        Parameters
        ----------
        dataset : DataFrame or Series containing texts of the corpus of size 'n_samples'.
            If dataset is DataFrame, the column containing texts will be identified
            from value of 'self.texts_column_name'.

        Returns
        -------
        T : DataFrame of shape (n_samples, n_topics)
            Returns the probabilities of each topic to be in the every given text.
            Topic's probabilities are ordered according to topics ordering in 'self.topics' property.
        """
        self._check_if_already_fitted()

        os.makedirs(self.working_dir_path, exist_ok=True)

        with tempfile.TemporaryDirectory(dir=self.working_dir_path) as extractor_working_dir:
            if PROCESSED_TEXT_COLUMN not in dataset.columns:
                process_dataset(
                    dataset,
                    self.texts_column_name,
                    extractor_working_dir,
                    **self.preprocessing_params
                )
                preprocessed_dataset = pd.read_csv(os.path.join(extractor_working_dir, PREPOCESSED_DATASET_FILENAME))
            else:
                preprocessed_dataset = dataset
            topics_extractor = TopicsExtractor(self._model)
            mixtures = topics_extractor.get_prob_mixture(
                dataset=preprocessed_dataset, working_dir=extractor_working_dir
            )

        return mixtures

    def fit_predict(self, dataset: Union[pd.DataFrame, pd.Series]) -> pd.DataFrame:
        """
        Preprocess texts in the datasets, looks for the best hyperparameters for ARTM model, fits the model
        with these parameters and predict topics mixtures for individual documents in the incoming corpus.
        The instance will contain topics with the most probable words belonging to them.

        Parameters
        ----------
        dataset : DataFrame or Series containing texts of the corpus of size 'n_samples'.
            If dataset is DataFrame, the column containing texts will be identified
            from value of 'self.texts_column_name'.

        Returns
        -------
        T : DataFrame of shape (n_samples, n_topics)
            Returns the probabilities of each topic to be in the every given text.
            Topic's probabilities are ordered according to topics ordering in 'self.topics' property.
        """
        self._check_if_already_fitted(fit_is_ok=False)

        processed_dataset_path = os.path.join(self.working_dir_path, f"{uuid.uuid4()}")
        self.fit(dataset, processed_dataset_path=processed_dataset_path)

        preprocessed_dataset = pd.read_csv(os.path.join(processed_dataset_path, PREPOCESSED_DATASET_FILENAME))
        return self.predict(preprocessed_dataset)

    def save(self, path: str, overwrite: bool = False):
        """
        Saves AutoTM to a filesystem.
        :param path: local filesystem path to save AutoTM on
        :param overwrite: if True and path: alredy exists, will try to remove path:
        """
        path_exists = os.path.exists(path)
        if path_exists and not overwrite:
            raise RuntimeError("The path is already exists and is not allowed to overwrite")
        elif path_exists:
            logger.debug(f"Removing existing path: {path}")
            shutil.rmtree(path)

        os.makedirs(path)

        artm_model_path = os.path.join(path, self._ARTM_MODEL_FILENAME)
        autotm_data_path = os.path.join(path, self._AUTOTM_DATA_FILENAME)

        self._model.dump_artm_model(artm_model_path)
        with open(autotm_data_path, "wb") as f:
            pickle.dump(self.get_params(), f)

    @property
    def topics(self) -> Dict[str, List[str]]:
        """
        Inferred set of topics with their respective top words.
        """
        self._check_if_already_fitted()
        return extract_topics(self._model)

    def print_topics(self):
        """
        Print topics in a human readable form in stdout
        """
        self._check_if_already_fitted()
        print_topics(self._model)

    def _check_if_already_fitted(self, fit_is_ok=True):
        if fit_is_ok and self._model is None:
            raise RuntimeError("The model is not fitted")

        if not fit_is_ok and self._model is not None:
            raise RuntimeError("The model is already fitted")
