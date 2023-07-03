from typing import Union, Optional, Any, Dict

import pandas as pd
from numpy.typing import ArrayLike

from autotm.infer import TopicsExtractor


class AutoTM:
    @classmethod
    def load(cls, path: str) -> 'AutoTM':
        pass

    def __init__(self,
                 topic_count: int = 10,
                 alg_name: str = "ga",
                 alg_params: Optional[Dict[str, Any]] = None,
                 surrogate_alg_name: Optional[str] = None,
                 surrogate_alg_params: Optional[Dict[str, Any]] = None,
                 artm_train_options: Optional[Dict[str, Any]] = None,
                 intermediate_tmp_files_path: str = '/tmp',
                 texts_column_name: str = "text",
                 log_file_path: Optional[str] = None,
                 exp_id: Optional[str] = None,
                 tag: Optional[str] = None,
                 ):
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

    def fit_predict(self, dataset: Union[pd.DataFrame, pd.Series, str]) -> pd.Series:
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
        pass

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
