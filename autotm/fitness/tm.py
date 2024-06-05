import logging
import multiprocessing
import multiprocessing as mp
import os
import pickle
import random
import time
import uuid
from collections import OrderedDict
from contextlib import contextmanager
from statistics import mean
from typing import Dict, Optional, Tuple, ContextManager, List, Callable, Union

import artm
import gensim.corpora as corpora
import numpy as np
import pandas as pd
from billiard.exceptions import SoftTimeLimitExceeded
from openai import OpenAI
from tqdm import tqdm

from autotm.abstract_params import AbstractParams
from autotm.batch_vect_utils import SampleBatchVectorizer
from autotm.fitness import AUTOTM_COMPONENT
from autotm.fitness.external_scores import ts_bground, ts_uniform, ts_vacuous, switchp
from autotm.utils import (
    MetricsScores,
    AVG_COHERENCE_SCORE,
    TimeMeasurements,
    log_exec_timer, LLM_SCORE, )

ENV_AUTOTM_LLM_API_KEY = "AUTOTM_LLM_API_KEY"
ENV_AUTOTM_LLM_MAX_ESTIMATED_TOPICS = "AUTOTM_LLM_MAX_ESTIMATED_TOPICS"
ENV_AUTOTM_LLM_ESTIMATIONS_PER_TOPIC = "AUTOTM_LLM_ESTIMATIONS_PER_TOPIC"


logger = logging.getLogger()

SYSTEM_PROMT_GPT4O_TOPICS_EVAL = """
You are a helpful assistant evaluating the top words of a topic model output for a given topic. 
Please rate how related the following words are to each other on a scale from 1 to 4 ("1" = poorly related, "2" = rather poorly related, "3" = rather related, "4" = very related). 
Reply with a single number, indicating the overall appropriateness of the topic.
"""


def estimate_topics_with_llm(model_id: str,
                             topics: Dict[str, List[str]],
                             api_key: str,
                             model_name: str = "gpt-4o",
                             num_top_words: int = 10,
                             max_estimated_topics: Optional[int] = None,
                             estimations_per_topic: int = 5,
                             seed: int = 42,
                             agg_func: Union[str, Callable[[Dict[str, float]], float]] = 'mean') -> float:
    """
    Estimates quality of the topics by calling ChatGPT and asking it to set the score for a topic.
    For every topic being estimated a separate call to the API is issued,
    because OpenAI doesn't support batch calls for chat completioncs
    :param model_name: Name of OpenAI model to use for scoring
    :param model_id: unique identifier of the model beingh evaluated
    :param topics: dictionary of topics (topic_name -> top words). Names of topics should follow 'main*' pattern or 'back*' pattern
    :param api_key: OpenAI key to execute queries.
    :param num_top_words: Number of first words in wordsets to be used for scoring.
    :param max_estimated_topics: Number of topics to be used for scoring. If this number is less that number of main topics than a random subset of topics will be sent for scoring.
    :param estimations_per_topic: Number of scoring repeatitions by ChatGPT to be conducted for every topic (the topic score is an average of all repeaatitions)
    :param seed: a random seed to use for sampling main topics
    :param agg_func: a function name or function to be used for final scores aggregation. If it is a name, only the following values are allowed: 'mean', 'min', 'max'
    :return: fitness score (float)
    """
    main_topics = {topic: words[:num_top_words] for topic, words in topics.items() if topic.startswith("main")}
    all_main_topics_count = len(main_topics)

    if max_estimated_topics and len(main_topics) > max_estimated_topics:
        keys = sorted(main_topics.keys())
        keys = random.Random(x=seed).sample(keys, k=max_estimated_topics)
        main_topics = {k: main_topics[k] for k in keys}

    if not api_key:
        raise ValueError(f"Api Key for ChatGPT is not provided. Probably, env var {ENV_AUTOTM_LLM_API_KEY} is not set.")

    agg_funcs = {
        'mean': lambda x: mean(x.values()) if len(x) > 0 else 0.0,
        'min': lambda x: min(x.values()) if len(x) > 0 else 0.0,
        'max': lambda x: max(x.values()) if len(x) > 0 else 0.0
    }
    if isinstance(agg_func, str) and agg_func not in agg_funcs:
        raise ValueError(
            f"Only the following aggregating functions available by name: {agg_funcs.keys()}. Received: {agg_func}"
        )
    elif isinstance(agg_func, str):
        agg_func = agg_funcs[agg_func]

    logger.info(
        "Estimating fitness (model %s) for %s main topics (of %s all main topics) "
        "with %s repetiton(s)" % (model_id, len(main_topics), all_main_topics_count, estimations_per_topic)
    )

    client = OpenAI(api_key=api_key)

    topics_scores = dict()
    estimations_counter = 0
    for topic, words in main_topics.items():
        user_prompt = ", ".join(words)

        individual_topic_scores = []
        for i in range(estimations_per_topic):
            chat_completion = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": SYSTEM_PROMT_GPT4O_TOPICS_EVAL},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=1.0,
                max_tokens=1
            )
            score = chat_completion.choices[0].message.content.strip()
            estimations_counter += 1
            try:
                score = int(score)
            except ValueError:
                logger.warning(
                    f"Not numeric score estimation happened (model %s, topic %s): '%s'. "
                    f"Words: %s" % (model_id, topic, score, words)
                )

            individual_topic_scores.append(score)

        topic_score = mean(individual_topic_scores) if len(individual_topic_scores) > 0 else 0.0
        topics_scores[topic] = topic_score

    fitness = agg_func(topics_scores)

    logger.debug(
        "Estimated fitness (model %s): %s. "
        "Number of estimations with ChatGPT: %s" % (model_id, fitness, estimations_counter)
    )

    return fitness


def extract_topics(model: artm.ARTM):
    if "TopTokensScore" not in model.score_tracker:
        logger.warning(
            "Key 'TopTokensScore' is not presented in the model's score_tracker. "
            "Returning empty dict of topics."
        )
        return dict()
    res = model.score_tracker["TopTokensScore"].last_tokens
    topics = {topic: tokens[:50] for topic, tokens in res.items()}
    return topics


def print_topics(model: artm.ARTM):
    for i, (topic, top_tokens) in enumerate(extract_topics(model).items()):
        logging.info(topic)
        logging.info(top_tokens)


class Dataset:
    _batches_path: str = "batches"
    _wv_path: str = "test_set_data_voc.txt"
    _cooc_dict_path: str = "cooc_dictionary.txt"
    _dictionary_path: str = "dictionary.txt"
    _vocab_path: str = "vocab.txt"
    _cooc_file_df_path: str = "cooc_df.txt"
    _cooc_file_tf_path: str = "cooc_tf.txt"
    _ppmi_dict_df_path: str = "ppmi_df.txt"
    _ppmi_dict_tf_path: str = "ppmi_tf.txt"
    _mutual_info_dict_path: str = "mutual_info_dict.pkl"
    _texts_path: str = "dataset_processed.csv"
    _labels_path = "labels.pkl"

    def __init__(self, base_path: str, topic_count: int):
        self._base_path = base_path
        self._topic_count = topic_count

        self._labels = None
        self._dictionary: Optional[artm.Dictionary] = None
        self._batches: Optional[artm.BatchVectorizer] = None
        self._mutual_info_dict: Optional[Dict] = None

    @property
    def base_path(self) -> str:
        """
        :return: The path to the directory that contains this dataset files
        """
        return self._base_path

    @property
    def topic_count(self) -> int:
        return self._topic_count

    @property
    def labels(self):
        return self._labels

    @property
    def dictionary(self) -> artm.Dictionary:
        assert self._dictionary
        return self._dictionary

    @property
    def batches(self) -> artm.BatchVectorizer:
        assert self._batches
        return self._batches

    @property
    def mutual_info_dict(self) -> Dict:
        assert self._mutual_info_dict
        return self._mutual_info_dict

    @property
    def sample_batches(self) -> SampleBatchVectorizer:
        batches_dir_path = self._make_path(self._batches_path)
        return SampleBatchVectorizer(data_path=batches_dir_path, data_format="batches")

    @property
    def texts(self) -> List[str]:
        """
        :return: a list of PROCESSED texts of the corpus
        """
        texts_df_path = self._make_path(self._texts_path)
        df = pd.read_csv(texts_df_path)
        return df["processed_text"].tolist()

    @property
    def total_tokens(self) -> int:
        wv_path = self._make_path(self._wv_path)
        with open(wv_path) as f:
            total_tokens = sum(
                (
                    sum([int(elem.split(":")[1]) for elem in line.split()[1:]])
                    for line in f
                )
            )
        return total_tokens

    def verify_dataset_files(self):
        """
        Verifies that mandatory files exist in the dataset folder defined by :py:func:'.base_path'
        """
        paths = [
            self._make_path(self._dictionary_path),
            self._make_path(self._batches_path),
            self._make_path(self._mutual_info_dict_path),
            self._make_path(self._texts_path),
        ]

        for p in paths:
            assert os.path.exists(
                p
            ), f"Dataset file on path {p} should exist, but it doesn't"

    def load_dataset(self):
        """
        Partially loads this dataset files creating supplementary data_generator structures and entities like artm.BatchVectorizer
        """
        logger.info("Loading dataset entities")

        self.verify_dataset_files()

        start = time.time()
        labels_path = self._make_path(self._labels_path)
        dictionary_path = self._make_path(self._dictionary_path)
        batches_dir_path = self._make_path(self._batches_path)
        mutual_info_dict_path = self._make_path(self._mutual_info_dict_path)

        if os.path.exists(labels_path):
            logger.info(f"Reading labels {labels_path}")
            with open(labels_path, "rb") as f:
                self._labels = pickle.load(f)

        logger.info(f"Reading dictionary from {dictionary_path}")
        self._dictionary = artm.Dictionary()
        self._dictionary.load_text(dictionary_path)
        # TODO: tune the params
        self._dictionary.filter(min_df=5, max_tf=1000)

        self._batches = artm.BatchVectorizer(
            data_path=batches_dir_path, data_format="batches"
        )

        with open(mutual_info_dict_path, "rb") as handle:
            self._mutual_info_dict = pickle.load(handle)

        logging.info(f"Dataset entities initialization took {time.time() - start: .2f}")

    def _make_path(self, path):
        """
        Creates a full path out of a relative path or a filename
        :param path: a relative path or a filename
        :return: a full path
        """
        return os.path.join(self._base_path, path)


class TopicModelFactory:
    num_processors: Optional[int] = mp.cpu_count()
    experiments_path: str = "/tmp/tm_experiments"
    cached_dataset_settings: Dict[str, Dataset] = dict()
    cached_datasets: Dict[Tuple, Dataset] = dict()

    @classmethod
    def init_factory_settings(
            cls,
            num_processors: Optional[int] = mp.cpu_count(),
            dataset_settings: Dict[str, Dict[str, object]] = None,
    ):
        cls.num_processors = num_processors
        cls.cached_dataset_settings = (
            {k: cls.init_dataset(v) for k, v in dataset_settings.items()}
            if dataset_settings
            else dict()
        )

    @classmethod
    def init_dataset(cls, settings) -> Dataset:
        assert "base_path" in settings and "topic_count" in settings
        dataset = Dataset(
            base_path=settings["base_path"], topic_count=settings["topic_count"]
        )
        dataset.load_dataset()
        return dataset

    def __init__(
            self,
            dataset_name: str,
            data_path: str,
            fitness_name: str,
            params: AbstractParams,
            topic_count: Optional[int] = None,
            forced_update: bool = False,
            train_option: str = "offline",
    ):
        self.dataset_name = dataset_name
        self.data_path = data_path
        self.fitness_name = fitness_name
        self.params = params
        self.topic_count = topic_count
        self.forced_update = forced_update
        self.train_option = train_option
        self._custom_scores = []
        self.tm = None

    def __enter__(self) -> "TopicModel":
        # local or cluster
        if AUTOTM_COMPONENT == "worker":
            if self.dataset_name not in self.cached_dataset_settings:
                raise Exception(f"No settings for dataset {self.dataset_name}")
            dataset = self.cached_dataset_settings[self.dataset_name]
            self.topic_count = self.topic_count if self.topic_count else dataset.topic_count
            logging.debug(f"Using the following settings: \n{dataset.base_path}")
        else:
            dataset = Dataset(base_path=self.data_path, topic_count=self.topic_count)
            if (self.data_path, self.topic_count) in self.cached_datasets:
                dataset = self.cached_datasets[(self.data_path, self.topic_count)]
            else:
                dataset.load_dataset()
                self.cached_datasets[(self.data_path, self.topic_count)] = dataset

        uid = uuid.uuid4()

        logging.info(
            f"Using TM model: {TopicModel} according "
            f"to fitness name: {self.fitness_name}, topics count: {self.topic_count}"
        )
        self.tm = TopicModel(
            uid,
            self.topic_count,
            self.num_processors,
            dataset,
            self.params,
            train_option=self.train_option,
        )

        self.tm.init_model()

        return self.tm

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.tm.dispose()


@contextmanager
def fit_tm_of_individual(
        dataset: str,
        data_path: str,
        params: AbstractParams,
        fitness_name: str = "default",
        topic_count: Optional[int] = None,
        force_dataset_settings_checkout: bool = False,
        train_option: str = "offline",
) -> ContextManager[Tuple[TimeMeasurements, MetricsScores, "TopicModel"]]:
    start = time.time()

    with TopicModelFactory(
            dataset,
            data_path,
            fitness_name,
            params,
            topic_count,
            force_dataset_settings_checkout,
            train_option,
    ) as tm:
        try:
            with log_exec_timer("TM Training") as train_timer:
                tm.train()

            if fitness_name in ["regular", "sparse"]:
                with log_exec_timer("Metrics calculation") as metrics_timer:
                    fitness = tm.metrics_get_last_avg_vals(
                        texts=tm.dataset.texts, total_tokens=tm.dataset.total_tokens
                    )
            elif fitness_name == "llm":
                with log_exec_timer("Metrics calculation") as metrics_timer:
                    fitness = estimate_topics_with_llm(
                        model_id=str(tm.uid),
                        topics=tm.get_topics(),
                        api_key=os.environ.get(ENV_AUTOTM_LLM_API_KEY, None),
                        max_estimated_topics=os.environ.get(ENV_AUTOTM_LLM_MAX_ESTIMATED_TOPICS, None),
                        estimations_per_topic=int(os.environ.get(ENV_AUTOTM_LLM_ESTIMATIONS_PER_TOPIC, "5"))
                    )
                fitness = {LLM_SCORE: fitness}
            else:
                raise ValueError(f"Unsupported fitness: {fitness_name}. "
                                 f"Only the following is supported: ['default', 'llm']")

            time_metrics = {
                "train": train_timer.duration,
                "metrics": metrics_timer.duration,
            }
        except SoftTimeLimitExceeded as ex:
            raise ex
        except Exception as e:
            logger.warning("Fitness calculation problem")
            fitness = {AVG_COHERENCE_SCORE: 0.0}
            time_metrics = {"train": -1, "metrics": -1}

        logging.info(f"Fitness estimation took {time.time() - start: .2f}")

        yield time_metrics, fitness, tm


class FitnessCalculatorWrapper:
    def __init__(self, dataset, data_path, topic_count, train_option):
        self.dataset = dataset
        self.data_path = data_path
        self.topic_count = topic_count
        self.train_option = train_option

    def run(self, params: AbstractParams):
        logging.info(params)
        # params = params[:-1] + [0, 0, 0] + [params[-1]]
        fitness = calculate_fitness_of_individual(
            dataset=self.dataset,
            data_path=self.data_path,
            params=params,
            topic_count=self.topic_count,
            train_option=self.train_option,
        )
        result = fitness[AVG_COHERENCE_SCORE]
        logging.info("Fitness: ", result)
        return -result


def calculate_fitness_of_individual(
        dataset: str,
        data_path: str,
        params: AbstractParams,
        fitness_name: str = "default",
        topic_count: Optional[int] = None,
        force_dataset_settings_checkout: bool = False,
        train_option: str = "offline",
) -> MetricsScores:
    with fit_tm_of_individual(
            dataset,
            data_path,
            params,
            fitness_name,
            topic_count,
            force_dataset_settings_checkout,
            train_option,
    ) as result:
        time_metrics, fitness, tm = result

    return fitness


class TopicModel:
    def __init__(
            self,
            uid: uuid.UUID,
            topic_count: int,
            num_processors: int,
            dataset: Dataset,
            params: AbstractParams,
            train_option: str = "offline",
    ):
        self.uid = uid
        self.topic_count: int = topic_count
        self.num_processors: int = num_processors
        self.dataset = dataset
        self.train_option = train_option

        self.model = None
        self.S = self.topic_count
        self.specific = ["main{}".format(i) for i in range(self.S)]

        self.params = params
        self.back = ["back{}".format(i) for i in range(int(self.params.basic_topics))]
        self._total_iterations = 0

    def init_model(self):
        self.model = artm.ARTM(
            num_topics=self.S + self.params.basic_topics,
            class_ids=["@default_class"],
            dictionary=self.dataset.dictionary,
            show_progress_bars=False,
            cache_theta=True,
            topic_names=self.specific + self.back,
            num_processors=self.num_processors,
        )

        self.__set_model_scores()

    def _early_stopping(self):
        coherences_main, coherences_back = self.__return_all_tokens_coherence(
            self.model, s=self.S, b=self.params.basic_topics
        )
        return len(coherences_main) < self.S or not any(coherences_main)

    def check_early_stop(self) -> bool:
        return self._total_iterations > 0 and self._early_stopping()

    def train(self):
        if self.model is None:
            logging.error("Initialise the model first!")
            return
        self._total_iterations = 0
        self.params.run_train(self)
        logging.info("Training is complete")

    def do_fit(self, n):
        n = int(n)
        self._total_iterations += n
        if self.train_option == "offline":
            self.model.fit_offline(
                batch_vectorizer=self.dataset.batches,
                num_collection_passes=n
            )
        elif self.train_option == "online_v1":
            self.model.fit_offline(
                batch_vectorizer=self.dataset.sample_batches,
                num_collection_passes=n,
            )
        elif self.train_option == "online_v2":
            self.model.num_document_passes = n
            self.model.fit_online(
                batch_vectorizer=self.dataset.sample_batches,
                update_every=self.num_processors,
            )
        else:
            raise ValueError(f"Unknown option {self.train_option}")

    def decor_train(self):
        if self.model is None:
            logging.error("Initialise the model first")
            return

        self.model.regularizers.add(
            artm.DecorrelatorPhiRegularizer(
                name="decorr", topic_names=self.specific, tau=self.decor
            )
        )

    def save_model(self, path):
        self.model.dump_artm_model(path)

    def print_topics(self):
        print_topics(self.model)

    def get_topics(self):
        return extract_topics(self.model)

    def _get_avg_coherence_score(self, for_individ_fitness=False):
        coherences_main, coherences_back = self.__return_all_tokens_coherence(
            self.model, s=self.S, b=self.params.basic_topics
        )
        if for_individ_fitness:
            return np.mean(list(coherences_main.values())) + np.min(
                list(coherences_main.values())
            )
        return np.mean(list(coherences_main.values()))

    # added
    def _calculate_labels_coeff(self, topk=3):
        theta = self.model.get_theta()
        # theta.set_index('Unnamed: 0', inplace=True)
        documents_topics = OrderedDict()
        topics_documents = OrderedDict()
        for col in theta.columns:
            topics = theta.nlargest(topk, col).index.tolist()
            documents_topics[col] = topics
            for tp in topics:
                if tp not in topics_documents:
                    topics_documents[tp] = []
                topics_documents[tp].append(col)
        total_res = []
        for label in self.dataset.labels:
            docs = self.dataset.labels[label]
            all_lens = []
            for topic in topics_documents:
                all_lens.append(
                    len(set(topics_documents[topic]).intersection(set(docs)))
                )
            total_res.append(max(all_lens) / len(docs))
        coeff = np.mean(total_res)
        logging.info(f"Calculated coefficient: {coeff}")
        return coeff

    def dispose(self):
        """
        Disposes the model instance and removes the model from the object making it reusable again
        :return:
        """
        self.model.dispose()
        self.model = None

        log_files = [file for file in os.listdir(".") if file.startswith("bigartm.")]
        logging.info(f"Deleting bigartm logs: {log_files}")
        for file in log_files:
            os.remove(file)

    def __set_model_scores(self):
        self.model.scores.add(
            artm.PerplexityScore(
                name="PerplexityScore", dictionary=self.dataset.dictionary
            )
        )

        self.model.scores.add(
            artm.SparsityPhiScore(
                name="SparsityPhiScore",
                class_id="@default_class",
                topic_names=self.specific,
            )
        )
        self.model.scores.add(
            artm.SparsityThetaScore(
                name="SparsityThetaScore", topic_names=self.specific
            )
        )

        # Fraction of background words in the whole collection
        self.model.scores.add(
            artm.BackgroundTokensRatioScore(
                name="BackgroundTokensRatioScore", class_id="@default_class"
            )
        )

        # Kernel characteristics
        self.model.scores.add(
            artm.TopicKernelScore(
                name="TopicKernelScore",
                class_id="@default_class",
                topic_names=self.specific,
                probability_mass_threshold=0.5,
                dictionary=self.dataset.dictionary,
            )
        )

        # Looking at top tokens
        self.model.scores.add(
            artm.TopTokensScore(
                name="TopTokensScore", class_id="@default_class", num_tokens=100
            )
        )

    def __calculate_topic_coherence(self, tokens, top=50):
        tokens = tokens[:top]
        total_sum = 0
        for ix, token_1 in enumerate(tokens[:-1]):
            for ij, token_2 in enumerate(tokens[(ix + 1):]):
                try:
                    total_sum += self.dataset.mutual_info_dict[
                        "{}_{}".format(token_1, token_2)
                    ]
                except KeyError:
                    total_sum += 0

        coherence = 2 / (top * (top - 1)) * total_sum
        return coherence

    def __return_all_tokens_coherence(self, model, s, b, top=50, return_backs=True):
        topics = list(model.score_tracker["TopTokensScore"].last_tokens.keys())

        res = model.score_tracker["TopTokensScore"].last_tokens

        topics_main = [topic for topic in topics if topic.startswith("main")]
        topics_back = [topic for topic in topics if topic.startswith("back")]

        all_topics_main = [i for i in range(s)]
        existing_topics_main = [int(i[4:]) for i in topics_main]
        inexisting_topics_main = [
            i for i in all_topics_main if i not in existing_topics_main
        ]

        all_topics_back = [i for i in range(b)]
        existing_topics_back = [int(i[4:]) for i in topics_back]
        inexisting_topics_back = [
            i for i in all_topics_back if i not in existing_topics_back
        ]

        coh_vals_main = {}
        # coherence for main topics
        for i, topic in tqdm(enumerate(topics_main), disable=True):
            coh_vals_main[topic] = self.__calculate_topic_coherence(
                res[topic][:50], top=top
            )
        for i, topic in tqdm(enumerate(inexisting_topics_main), disable=True):
            coh_vals_main["main{}".format(i)] = 0

        coh_vals_back = {}
        # coherence for back topics
        for i, topic in tqdm(enumerate(topics_back), disable=True):
            coh_vals_back[topic] = self.__calculate_topic_coherence(
                res[topic][:50], top=top
            )
        for i, topic in tqdm(enumerate(inexisting_topics_back), disable=True):
            coh_vals_back["back{}".format(i)] = 10  # penalty for not creating backs

        if return_backs:
            return coh_vals_main, coh_vals_back
        else:
            return coh_vals_main

    # TODO: fix
    def __return_all_coherence_types(
            self, model, S, only_specific=True, top=(10, 15, 20, 25, 30, 35, 40, 45, 50, 55)
    ):
        topics = list(model.score_tracker["TopTokensScore"].last_tokens.keys())
        coh_vals = {}

        res = model.score_tracker["TopTokensScore"].last_tokens
        if only_specific:
            topics = [topic for topic in topics if topic.startswith("main")]

        all_topics = [i for i in range(S)]
        existing_topics = [int(i[4:]) for i in topics]
        inexisting_topics = [i for i in all_topics if i not in existing_topics]

        for num_tokens in top:
            coh_vals["coherence_{}".format(num_tokens)] = {}
            for i, topic in tqdm(enumerate(topics), disable=True):
                coh_vals["coherence_{}".format(num_tokens)][
                    topic
                ] = self.__calculate_topic_coherence(res[topic][:100], top=num_tokens)
            for i, topic in tqdm(enumerate(inexisting_topics), disable=True):
                coh_vals["coherence_{}".format(num_tokens)][topic] = 0
        return coh_vals

    def metrics_get_avg_coherence_score(
            self, for_individ_fitness=False
    ) -> MetricsScores:
        coherences_main, coherences_back = self.__return_all_tokens_coherence(
            self.model, s=self.S, b=self.params.basic_topics
        )
        # commented after Masha's consultation
        # coeff = self._calculate_labels_coeff()
        coeff = 1.0
        if for_individ_fitness:
            logging.info(f"COMPONENTS {np.mean(list(coherences_main.values()))} {np.min(list(coherences_main.values()))}")
            avg_coherence_score = (
                    np.mean(list(coherences_main.values())) + np.min(list(coherences_main.values())) * coeff
            )
        else:
            avg_coherence_score = np.mean(list(coherences_main.values())) * coeff

        return {AVG_COHERENCE_SCORE: avg_coherence_score}

    # TODO: fix
    def metrics_get_last_avg_vals(
            self,
            texts,
            total_tokens,
            calculate_significance=False,
            calculate_npmi=False,
            calculate_switchp=False,
    ) -> MetricsScores:
        if calculate_significance:
            # turn off significance calculation
            phi_matrix = self.model.get_phi()
            theta_matrix = self.model.get_theta()

            phi_matrix = phi_matrix.to_numpy()
            theta_matrix = theta_matrix.to_numpy()
            topic_word_dist = phi_matrix[:, : self.S]  # getting subject scores
            doc_topic_dist = theta_matrix[: self.S, :]  #

            topic_significance_uni = np.mean(ts_uniform(topic_word_dist))
            topic_significance_vacuous = np.mean(
                ts_vacuous(doc_topic_dist, topic_word_dist, total_tokens)
            )
            topic_significance_back = np.mean(ts_bground(doc_topic_dist))
            logging.info(
                f"Topic Significance - Uniform Distribution Over Words: {topic_significance_uni}"
            )
            logging.info(
                f"Topic Significance - Vacuous Semantic Distribution: {topic_significance_vacuous}"
            )
            logging.info(
                f"Topic Significance - Background Distribution: {topic_significance_back}"
            )
        else:
            topic_significance_uni = None
            topic_significance_vacuous = None
            topic_significance_back = None

        topic_names = list(
            self.model.score_tracker["TopTokensScore"].last_tokens.keys()
        )
        topics_dict = self.model.score_tracker["TopTokensScore"].last_tokens

        specific_topics = [topic for topic in topic_names if topic.startswith("main")]

        all_topics_flag = False
        if len(specific_topics) == self.S:
            logging.info("Wow! all topics")
            all_topics_flag = True

        logger.info("Building dictionary")

        if calculate_npmi:
            splitted_texts = [text.split() for text in texts]
            id2word = corpora.Dictionary(splitted_texts)

            logger.info("Calculating NPMIs")

            npmis = (
                (
                    num_tokens,
                    calculate_npmi(splitted_texts, id2word, topics_dict, num_tokens),
                )
                for num_tokens in [15, 25, 50]
            )

            npmis = {
                k: v
                for num_tokens, npmi_list in npmis
                for k, v in [
                    (f"npmi_{num_tokens}", np.mean(npmi_list) if npmi_list else None),
                    (f"npmi_{num_tokens}_list", npmi_list),
                ]
            }

        else:
            npmis = {
                "npmi_50": None,
                "npmi_15": None,
                "npmi_25": None,
                "npmi_50_list": None,
            }

        if calculate_switchp:
            logger.info("Calculating switchp")

            if texts:
                switchp_scores = switchp(self.model.get_phi(), texts)
                switchp_scores = [
                    score if score is not None else 0 for score in switchp_scores
                ]
                logging.info(f"SwitchP mean: {np.mean(switchp_scores)}")
            else:
                switchp_scores = None
        else:
            switchp_scores = [np.nan]

        background_ratio = self.model.score_tracker[
            "BackgroundTokensRatioScore"
        ].last_value
        control_perplexity = self.model.score_tracker["PerplexityScore"].last_value
        sparsity_phi = self.model.score_tracker["SparsityPhiScore"].last_value
        sparsity_theta = self.model.score_tracker["SparsityThetaScore"].last_value
        contrast = self.model.score_tracker["TopicKernelScore"].last_average_contrast
        purity = self.model.score_tracker["TopicKernelScore"].last_average_purity
        kernel_size = self.model.score_tracker["TopicKernelScore"].last_average_size

        avg_coherence_score = self.metrics_get_avg_coherence_score(
            for_individ_fitness=True
        )
        coherences = self.__return_all_coherence_types(
            self.model, self.S, only_specific=True
        )

        coherence_scores = dict()
        for i in range(10, 60, 5):
            coherence = list(coherences[f"coherence_{i}"].values())
            coherence_scores.update(
                {f"coherence_{i}_list": coherence, f"coherence_{i}": np.mean(coherence)}
            )

        scores_dict = {
            AVG_COHERENCE_SCORE: avg_coherence_score[AVG_COHERENCE_SCORE],
            "perplexityScore": control_perplexity,
            "backgroundTokensRatioScore": background_ratio,
            "contrast": contrast,
            "purity": purity,
            "kernelSize": kernel_size,
            "npmi_50_list": npmis["npmi_50_list"],  # npmi_values_50_list,
            "npmi_50": npmis["npmi_50"],
            "sparsity_phi": sparsity_phi,
            "sparsity_theta": sparsity_theta,
            "topic_significance_uni": topic_significance_uni,
            "topic_significance_vacuous": topic_significance_vacuous,
            "topic_significance_back": topic_significance_back,
            "switchP_list": switchp_scores,
            "switchP": np.nan if np.isnan(switchp_scores).all() else np.nanmean(switchp_scores),
            "all_topics": all_topics_flag,
            **coherence_scores,
            **npmis,
        }

        return scores_dict


def fit_tm(preproc_data_path: str, topic_count: int, params: AbstractParams, train_option: str) -> TopicModel:
    with log_exec_timer("Loading dataset: "):
        dataset = Dataset(base_path=preproc_data_path, topic_count=topic_count)
        dataset.load_dataset()

    tm = TopicModel(
        uuid.uuid4(),
        topic_count,
        num_processors=multiprocessing.cpu_count(),
        dataset=dataset,
        params=params,
        train_option=train_option,
    )

    tm.init_model()

    with log_exec_timer("TM Training"):
        tm.train()

    return tm
