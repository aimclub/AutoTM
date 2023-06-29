# source: https://github.com/machine-intelligence-laboratory/OptimalNumberOfTopics/blob/master/topnum/scores/intratext_coherence_score.py

import dill
import numpy as np
import pandas as pd
import sys
import tqdm
import warnings

from collections import defaultdict
from enum import Enum, IntEnum, auto
from typing import (
    Callable,
    Dict,
    List,
    Optional,
    Tuple,
    Union,
)

from base_score import BaseScore

VW_TEXT_COL = "vw_text"
RAW_TEXT_COL = "raw_text"


class TextType(Enum):
    VW_TEXT = VW_TEXT_COL
    RAW_TEXT = RAW_TEXT_COL


class ComputationMethod(IntEnum):
    """
    Ways to compute intra-text coherence
    (see more about coherence below in IntratextCoherenceScore)
    Attributes
    ----------
    SEGMENT_LENGTH :
        Estimate the length of topic segments
    SEGMENT_WEIGHT :
        Estimate the weight of topic segment
        (weight - sum of specificities for the topic over words in segment)
    SUM_OVER_WINDOW :
        Sum of specificities for the topic over words in given window.
        The process is as follows:
        word of the topic is found in text, it is the center of the first window;
        next word of the topic is found (outside of the previous window), window; etc
    """

    SEGMENT_LENGTH = auto()
    SEGMENT_WEIGHT = auto()
    SUM_OVER_WINDOW = auto()


class WordTopicRelatednessType(IntEnum):
    """
    Word-topic relatedness estimate
    Attributes
    ----------
    PWT :
        p(w | t)
    PTW :
        p(t | w)
    """

    PWT = auto()
    PTW = auto()


class SpecificityEstimationMethod(IntEnum):
    """
    Way to estimate how particular word is specific for particular topic.
    Unlike probability, eg. p(w | t), specificity_estimation takes into account
    values for all topics, eg. p(w | t_1), p(w | t_2), ..., p(w | t_n):
    the higher the value p(w | t) comparing other p(w | t_i),
    the higher the specificity_estimation of word "w" for the topic "t"
    Attributes
    ----------
    NONE :
        Don't try to estimate specificity_estimation, return the probability as is
    MAXIMUM :
        From probability, corresponding to word and topic,
        extract *maximum* among probabilities for the word and other topics
    AVERAGE :
        From probability, corresponding to word and topic,
        extract *average* among probabilities for the word and other topics
    """

    NONE = auto()
    MAXIMUM = auto()
    AVERAGE = auto()


class IntratextCoherenceScore(BaseScore):
    """Computes intratext coherence
    For each topic of topic model its distribution throughout document collection is observed.
    Hypothetically, the better the topic, the more often it is represented by
    long segments of words highly related to the topic.
    The score tries to bring to life this idea.
    For more details one may see the article http://www.dialog-21.ru/media/4281/alekseevva.pdf
    """

    def __init__(  # noqa: C901
        self,
        dataset: Union[Dataset, str],
        name: str = None,
        should_compute: Callable[[int], bool] = None,
        keep_dataset_in_memory: bool = None,
        keep_dataset: bool = True,
        documents: List[str] = None,
        documents_fraction: float = 1.0,
        text_type: TextType = TextType.VW_TEXT,
        computation_method: ComputationMethod = ComputationMethod.SEGMENT_WEIGHT,
        word_topic_relatedness: WordTopicRelatednessType = WordTopicRelatednessType.PWT,
        specificity_estimation: SpecificityEstimationMethod = SpecificityEstimationMethod.NONE,
        max_num_out_of_topic_words: int = 10,
        window: int = 20,
        start_fit_iteration: int = 0,
        fit_iteration_step: int = 1,
        seed: int = 11221963,
        verbose: bool = False,
    ):
        """
        Parameters
        ----------
        name:
            Name of the score
        dataset : Dataset
            Dataset with document collection, or path to dataset
            (any model passed to `call()` is supposed to be trained on it)
        keep_dataset_in_memory
            Whether to keep `dataset` in memory or not
            (parameter `_small_data` of the `dataset` object).
            If `dataset` is given as object of type `Dataset` (and not as `str` path to dataset),
            the parameter will be set equal to `dataset._small_data`.
            Otherwise, the default value is `True` and `dataset._small_data` will be overwritten.
        keep_dataset
            Whether to keep `dataset` constantly as inner part of the score,
            or recreate it for each `call()` invocation and then dispose
        documents : list of str
            Which documents from the dataset are to be used for computing coherence
        documents_fraction
            The fraction of all the documents in the Dataset to be used for coherence computation
            if `documents` parameter is not specified
        text_type : TextType
            What text to use when computing coherence: raw text or VW text
            Preferable to use VW (as it is usually preprocessed, stop-words removed etc.),
            and with words in *natural order*.
            Score needs "real" text to compute coherence
        computation_method : ComputationMethod
            The way to compute intra-text coherence
        word_topic_relatedness : WordTopicRelatednessType
            How to estimate word relevance to topic: using p(w | t) or p(t | w)
        specificity_estimation : SpecificityEstimationMethod
            How to estimate specificity of word to topic
        max_num_out_of_topic_words : int
            In case computation_method = ComputationMethod.SEGMENT_LENGTH or
            ComputationMethod.SEGMENT_WEIGHT:
            Maximum number of words not of the topic which can be encountered without stopping
            the process of adding words to the current segment
        window : int
            In case computation_method = ComputationMethod.SUM_OVER_WINDOW:
            Window width. So the window will be the words with positions
            in [current position - window / 2, current position + window / 2)
        start_fit_iteration
            Indicates how many calls are skipped before the actual score is calculated.
            Replaces not calculated values with placeholders
            (for consistency of score values with number of model fit iterations).
        fit_iteration_step
            Number of iterations between `score.call()` invocations which actually update the score
        seed
            Random seed used for documents subsampling if `documents` parameter is not specified
        Notes
        -----
        Parameters `start_fit_iteration` and `fit_iteration_step` are introduced
        to reduce the time needed for one model training.
        If one is interested only in the last score value
        at the end of the training process (and not in the dependence of score on iteration),
        one should adjust `start_fit_iteration` and `fit_iteration_step` correspondingly.
        For example:
        >>> # dataset = Dataset(...)
        >>> # topic_model = TopicModel(...)
        >>> num_iterations = 100
        >>> topic_model.custom_scores['intratext_coherence'] = IntratextCoherenceScore(
        >>>     dataset,
        >>>     start_fit_iteration=num_iterations - 1  # last iteration: starting from zero
        >>> )
        >>> topic_model._fit(dataset.get_batch_vectorizer(), num_iterations=num_iterations)
        """
        # TODO: word_topic_relatedness seems to be connected with TopTokensViewer stuff
        super().__init__(name=name, should_compute=should_compute)

        self._keep_dataset = keep_dataset

        if isinstance(dataset, str):
            if keep_dataset_in_memory is None:
                keep_dataset_in_memory = True

            dataset = Dataset(data_path=dataset, keep_in_memory=keep_dataset_in_memory)

        self._keep_dataset_in_memory = dataset._small_data

        if not isinstance(dataset, Dataset):
            raise TypeError(
                f'Got "{type(dataset)}" as "dataset". Expect it to derive from "Dataset"'
            )

        if not isinstance(text_type, TextType):
            raise TypeError(
                f'Wrong "text_type": "{text_type}". ' f'Expect to be "{TextType}"'
            )

        if not isinstance(computation_method, ComputationMethod):
            raise TypeError(
                f'Wrong "computation_method": "{computation_method}". '
                f'Expect to be "{ComputationMethod}"'
            )

        if not isinstance(word_topic_relatedness, WordTopicRelatednessType):
            raise TypeError(
                f'Wrong "word_topic_relatedness": "{word_topic_relatedness}". '
                f'Expect to be "{WordTopicRelatednessType}"'
            )

        if not isinstance(specificity_estimation, SpecificityEstimationMethod):
            raise TypeError(
                f'Wrong "specificity_estimation": "{specificity_estimation}". '
                f'Expect to be "{SpecificityEstimationMethod}"'
            )

        if not isinstance(max_num_out_of_topic_words, int):
            raise TypeError(
                f'Wrong "max_num_out_of_topic_words": "{max_num_out_of_topic_words}". '
                f'Expect to be "int"'
            )

        if not isinstance(window, int):
            raise TypeError(f'Wrong "window": "{window}". Expect to be "int"')

        if window < 0 or (
            window == 0 and computation_method == ComputationMethod.SUM_OVER_WINDOW
        ):
            raise ValueError(
                f'Wrong value for "window": "{window}". '
                f"Expect to be non-negative. And greater than zero in case "
                f"computation_method == ComputationMethod.SUM_OVER_WINDOW"
            )

        if not isinstance(start_fit_iteration, int):
            raise TypeError(
                f'Wrong "start_fit_iteration": "{start_fit_iteration}".'
                f' Expect to be "int"'
            )

        if not isinstance(fit_iteration_step, int):
            raise TypeError(
                f'Wrong "fit_iteration_step": "{start_fit_iteration}".'
                f' Expect to be "int"'
            )
        if fit_iteration_step <= 0:
            raise ValueError(
                f'Wrong "fit_iteration_step": "{fit_iteration_step}".'
                f" Expect to be > 0"
            )

        if documents_fraction <= 0:
            raise ValueError(
                f'Wrong "documents_fraction": "{documents_fraction}".'
                f" Expect to be in (0, 1]"
            )
        if documents_fraction > 1.0:
            warnings.warn(
                f"Parameter documents_fraction={documents_fraction} can't be bigger than 1.0"
                f" Setting it equal to 1.0"
            )

            documents_fraction = 1.0

        self._dataset = dataset
        self._dataset_file_path = dataset._data_path
        self._dataset_internals_folder_path = dataset._internals_folder_path

        self._text_type = text_type
        self._computation_method = computation_method
        self._word_topic_relatedness = word_topic_relatedness
        self._specificity_estimation_method = specificity_estimation
        self._max_num_out_of_topic_words = max_num_out_of_topic_words
        self._window = window

        self._verbose = verbose

        self._current_iteration = 0
        self._start_fit_iteration = start_fit_iteration
        self._fit_iteration_step = fit_iteration_step

        if documents is not None:
            self._documents = documents
        else:
            all_documents = list(self._dataset.get_dataset().index)
            documents_fraction = min(documents_fraction, 1.0)
            num_documents_to_choose = int(
                np.ceil(len(all_documents) * documents_fraction)
            )
            custom_random = np.random.RandomState(seed)

            self._documents = list(
                custom_random.choice(
                    all_documents, size=num_documents_to_choose, replace=False
                )
            )

    def __repr__(self):
        return (
            f"{self.__class__.__name__}("
            f"text_type={self._text_type!r}"
            f"computation_method={self._computation_method!r}"
            f"word_topic_relatedness={self._word_topic_relatedness!r}"
            f"specificity_estimation_method={self._specificity_estimation_method!r}"
            f"max_num_out_of_topic_words={self._max_num_out_of_topic_words!r}"
            f"window={self._window!r}"
            f")"
        )

    @property
    def dataset(self) -> Dataset:
        return self._dataset

    @dataset.setter
    def dataset(self, new_dataset: Dataset) -> None:
        self._dataset = new_dataset
        self._dataset_file_path = new_dataset._data_path
        self._dataset_internals_folder_path = new_dataset._internals_folder_path
        self._keep_dataset_in_memory = new_dataset._small_data

    def save(self, path: str) -> None:
        dataset = self._dataset
        self._dataset = None

        with open(path, "wb") as f:
            dill.dump(self, f)

        self._dataset = dataset

    @classmethod
    def load(cls, path: str):
        """
        Parameters
        ----------
        path
        Returns
        -------
        IntratextCoherenceScore
        """
        score: IntratextCoherenceScore

        with open(path, "rb") as f:
            score = dill.load(f)

        if not score._keep_dataset:
            score._dataset = None
        else:
            score._dataset = Dataset(
                score._dataset_file_path,
                internals_folder_path=score._dataset_internals_folder_path,
                keep_in_memory=score._keep_dataset_in_memory,
            )

        return score

    def call(self, model: BaseModel, **kwargs) -> float:
        if (
            self._current_iteration - self._start_fit_iteration
        ) % self._fit_iteration_step != 0:
            self._current_iteration += 1

            return float("nan")

        try:
            if self._dataset is None:
                self._dataset = Dataset(
                    self._dataset_file_path,
                    internals_folder_path=self._dataset_internals_folder_path,
                    keep_in_memory=self._keep_dataset_in_memory,
                )

            topic_coherences = self.compute(model, None)

            coherence_values = list(
                v if v is not None else 0.0  # TODO: state the behavior clearer somehow
                for v in topic_coherences.values()
            )

            self._current_iteration += 1

            return float(np.median(coherence_values))  # TODO: or mean?

        finally:
            if not self._keep_dataset:
                self._dataset = None

    def compute(
        self, model: BaseModel, topics: List[str] = None, documents: List[str] = None
    ) -> Dict[str, Optional[float]]:
        if not isinstance(model, BaseModel):
            raise TypeError(
                f'Got "{type(model)}" as "model". '
                f'Expect it to derive from "BaseModel"'
            )

        if topics is None:
            topics = IntratextCoherenceScore._get_topics(model)

        if documents is None:
            documents = list(self._documents)

        if not isinstance(topics, list):
            raise TypeError(
                f'Got "{type(topics)}" as "topics". Expect list of topic names'
            )

        if not isinstance(documents, list):
            raise TypeError(
                f'Got "{type(documents)}" as "documents". Expect list of document ids'
            )

        word_topic_relatednesses = self._get_word_topic_relatednesses(model)

        topic_document_coherences = np.zeros((len(topics), len(documents)))
        document_indices_with_topic_coherence = defaultdict(list)

        if not self._verbose:
            document_enumeration = enumerate(documents)
        else:
            document_enumeration = tqdm.tqdm(
                enumerate(documents), total=len(documents), file=sys.stdout
            )

        for document_index, document in document_enumeration:
            for topic_index, topic in enumerate(topics):
                # TODO: read document text only once for all topics
                topic_coherence = self._compute_coherence(
                    topic, document, word_topic_relatednesses
                )

                if topic_coherence is not None:
                    topic_document_coherences[
                        topic_index, document_index
                    ] = topic_coherence
                    document_indices_with_topic_coherence[topic].append(document_index)

        topic_coherences = [
            topic_document_coherences[
                topic_index, document_indices_with_topic_coherence[topic]
            ]
            if len(document_indices_with_topic_coherence) > 0
            else list()
            for topic_index, topic in enumerate(topics)
        ]

        return dict(
            zip(
                topics,
                [
                    float(np.mean(coherence_values))
                    if len(coherence_values) > 0
                    else None
                    for coherence_values in topic_coherences
                ],
            )
        )

    @staticmethod
    def _get_topics(model):
        return list(model.get_phi().columns)

    def _get_word_topic_relatednesses(self, model) -> pd.DataFrame:
        phi = model.get_phi()

        word_topic_probs = self._get_word_topic_probs(phi)

        if self._specificity_estimation_method == SpecificityEstimationMethod.NONE:
            pass

        elif self._specificity_estimation_method == SpecificityEstimationMethod.AVERAGE:
            word_topic_probs[:] = word_topic_probs.values - np.sum(
                word_topic_probs.values, axis=1, keepdims=True
            ) / max(  # noqa E131
                word_topic_probs.shape[1], 1
            )  # noqa E131

        elif self._specificity_estimation_method == SpecificityEstimationMethod.MAXIMUM:
            new_columns = []

            for t in word_topic_probs.columns:
                new_column = word_topic_probs[t].values - np.max(
                    word_topic_probs[word_topic_probs.columns.difference([t])].values,
                    axis=1,
                )
                new_columns.append(list(new_column))

            word_topic_probs[:] = np.array(new_columns).T

        return word_topic_probs

    def _get_word_topic_probs(self, phi: pd.DataFrame) -> pd.DataFrame:
        if self._word_topic_relatedness == WordTopicRelatednessType.PWT:
            return phi

        elif self._word_topic_relatedness == WordTopicRelatednessType.PTW:
            # Treat all topics as equally probable
            eps = np.finfo(float).tiny

            pwt = phi
            pwt_values = pwt.values

            return pd.DataFrame(
                index=pwt.index,
                columns=pwt.columns,
                data=pwt_values / (pwt_values.sum(axis=1).reshape(-1, 1) + eps),
            )

        assert False

    def _compute_coherence(self, topic, document, word_topic_relatednesses):
        assert isinstance(self._computation_method, ComputationMethod)

        words = self._get_words(document)

        if self._computation_method == ComputationMethod.SUM_OVER_WINDOW:
            average_sum_over_window = self._sum_relatednesses_over_window(
                topic, words, word_topic_relatednesses
            )

            return average_sum_over_window

        (
            topic_segment_length,
            topic_segment_weight,
        ) = self._compute_segment_characteristics(
            topic, words, word_topic_relatednesses
        )

        if self._computation_method == ComputationMethod.SEGMENT_LENGTH:
            return topic_segment_length

        elif self._computation_method == ComputationMethod.SEGMENT_WEIGHT:
            return topic_segment_weight

    def _get_words(self, document):
        def get_biggest_modality_or_default():
            modalities = list(self._dataset.get_possible_modalities())

            if len(modalities) == 0:
                return DEFAULT_ARTM_MODALITY

            modalities_vocabulary_sizes = list(
                map(lambda m: self._dataset.get_dataset().loc[m].shape[0], modalities)
            )

            return modalities[np.argmax(modalities_vocabulary_sizes)]

        if self._text_type == TextType.RAW_TEXT:
            text = self._dataset.get_source_document(document).values[
                0, 0
            ]  # TODO: this way?
            modality = get_biggest_modality_or_default()

            return list(map(lambda w: (modality, w), text.split()))

        if self._text_type == TextType.VW_TEXT:
            text = self._dataset.get_vw_document(document).values[
                0, 0
            ]  # TODO: this way?

            words = []
            modality = None

            # TODO: there was similar bunch of code somewhere...
            for word in text.split()[1:]:  # skip document id
                if word.startswith(MODALITY_START_SYMBOL):
                    modality = word[1:]

                    continue

                word = word.split(":")[0]

                if modality is not None:
                    word = (modality, word)  # phi multiIndex
                else:
                    word = (DEFAULT_ARTM_MODALITY, word)

                words.append(word)

            return words

        assert False

    def _compute_segment_characteristics(
        self, topic, words, word_topic_relatednesses: pd.DataFrame
    ) -> Tuple[float, float]:
        topic_segment_lengths = []
        topic_segment_weights = []

        topic_index = word_topic_relatednesses.columns.get_loc(topic)
        word_topic_indices = np.argmax(word_topic_relatednesses.values, axis=1)

        def get_word_topic_index(word):
            if word not in word_topic_relatednesses.index:
                return -1
            else:
                return word_topic_indices[word_topic_relatednesses.index.get_loc(word)]

        index = 0

        while index < len(words):
            original_index = index

            if get_word_topic_index(words[index]) != topic_index:
                index += 1

                continue

            segment_length = 1
            segment_weight = IntratextCoherenceScore._get_relatedness(
                words[index], topic, word_topic_relatednesses
            )

            num_out_of_topic_words = 0

            index += 1

            while (
                index < len(words)
                and num_out_of_topic_words < self._max_num_out_of_topic_words
            ):
                if get_word_topic_index(words[index]) != topic_index:
                    num_out_of_topic_words += 1
                else:
                    segment_length += 1
                    segment_weight += IntratextCoherenceScore._get_relatedness(
                        words[index], topic, word_topic_relatednesses
                    )

                    num_out_of_topic_words = 0

                index += 1

            topic_segment_lengths.append(segment_length)
            topic_segment_weights.append(segment_weight)

            assert index > original_index

        if len(topic_segment_lengths) == 0:
            return None, None
        else:
            return np.mean(topic_segment_lengths), np.mean(topic_segment_weights)

    def _sum_relatednesses_over_window(
        self, topic, words, word_topic_relatednesses
    ) -> float:
        topic_index = word_topic_relatednesses.columns.get_loc(topic)
        word_topic_indices = np.argmax(word_topic_relatednesses.values, axis=1)

        def get_word_topic_index(word):
            if word not in word_topic_relatednesses.index:
                return -1
            else:
                return word_topic_indices[word_topic_relatednesses.index.get_loc(word)]

        def find_next_topic_word(starting_index: int) -> int:
            index = starting_index

            while (
                index < len(words) and get_word_topic_index(words[index]) != topic_index
            ):
                index += 1

            if index == len(words):
                return -1  # failed to find next topic word

            return index

        word_index = find_next_topic_word(0)

        if word_index == -1:
            return None

        sums = list()

        while word_index < len(words) and word_index != -1:
            original_word_index = word_index

            window_lower_bound = word_index - int(np.floor(self._window // 2))
            window_upper_bound = word_index + int(np.ceil(self._window // 2))

            sum_in_window = np.sum(
                [
                    IntratextCoherenceScore._get_relatedness(
                        w, topic, word_topic_relatednesses
                    )
                    for w in words[window_lower_bound:window_upper_bound]
                ]
            )

            sums.append(sum_in_window)

            word_index = find_next_topic_word(window_upper_bound)

            assert word_index > original_word_index or word_index == -1

        return np.mean(sums)

    @staticmethod
    def _get_relatedness(word, topic, word_topic_relatednesses: pd.DataFrame) -> float:
        if word in word_topic_relatednesses.index:
            return word_topic_relatednesses.loc[word, topic]

        # TODO: throw Warning or log somewhere?
        return np.mean(word_topic_relatednesses.values)
