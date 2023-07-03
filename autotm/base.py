from typing import Union, Optional, Any, Dict

import pandas as pd
from numpy.typing import ArrayLike
from sklearn.base import BaseEstimator

from autotm.infer import TopicsExtractor


class AutoTM(BaseEstimator):
    @classmethod
    def load(cls, path: str) -> 'AutoTM':
        """
        Loads AutoTM instance from a path on local filesystem.
        :param path: a local filesystem path to load an AutoTM instance from.
        """
        raise NotImplementedError()

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
        pass

    def fit(self, dataset: Union[pd.DataFrame, pd.Series]) -> 'AutoTM':
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

        """
        raise NotImplementedError()

    def predict(self, dataset: Union[pd.DataFrame, pd.Series, str]) -> ArrayLike:
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
        T : array-like of shape (n_samples, n_topics)
            Returns the probabilities of each topic to be in the every given text.
            Topic's probabilities are ordered according to topics ordering in 'self.topics' property.
        """
        raise NotImplementedError()

    def fit_predict(self, dataset: Union[pd.DataFrame, pd.Series, str]) -> ArrayLike:
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
        T : array-like of shape (n_samples, n_topics)
            Returns the probabilities of each topic to be in the every given text.
            Topic's probabilities are ordered according to topics ordering in 'self.topics' property.
        """
        raise NotImplementedError()

    def save(self, path: str):
        """
        Saves AutoTM to a filesystem.
        :param path: local filesystem path to save AutoTM on
        """
        raise NotImplementedError()

    @property
    def text_preprocessor(self):
        """
        An object responsible for text preprocessing before applying ARTM model.
        """
        raise NotImplementedError()

    @property
    def topics_extractor(self) -> TopicsExtractor:
        """
        An object responsible for topics mixture identification for texts in the incoming preprocessed dataset.
        """
        raise NotImplementedError()

    @property
    def topics(self) -> pd.DataFrame:
        """
        Inferred set of topics with their respective top words.
        """
        raise NotImplementedError()

    def print_topics(self):
        """
        Print topics in a human readable form in stdout
        """
        raise NotImplementedError()
