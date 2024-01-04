from __future__ import print_function

import glob
import logging
import os
import sys
import time
from dataclasses import dataclass
from typing import Dict, Tuple, List, Set

import artm
from six.moves import range

from autotm.preprocessing import RESERVED_TUPLE

logger = logging.getLogger(__name__)


@dataclass
class CoocDictionaries:
    cooc_df: Dict[Tuple[str, str], float]
    cooc_tf: Dict[Tuple[str, str], float]
    cooc_df_term: Dict[str, float]
    cooc_tf_term: Dict[str, float]


def __create_batch_dictionary(batch):
    batch_dictionary = {}
    for index, token in enumerate(batch.token):
        batch_dictionary[index] = token

    return batch_dictionary


def __process_batch(
        global_cooc_df_dictionary,
        global_cooc_tf_dictionary,
        global_cooc_df_term_dictionary,
        global_cooc_tf_term_dictionary,
        batch,
        window_size,
        vocab: Set[str]):

    global_cooc_tf_dictionary[RESERVED_TUPLE] = 0.0
    batch_dictionary = __create_batch_dictionary(batch)

    def __process_window_df(global_cooc_dict, global_word_dict, token_ids,
                            doc_seen_pairs: Set[Tuple[int, int]], doc_seen_words: Set[int]):
        for j in range(1, len(token_ids)):
            token_1 = batch_dictionary[token_ids[0]]
            token_2 = batch_dictionary[token_ids[j]]

            if token_1 == token_2:
                continue

            if token_1 not in vocab or token_2 not in vocab:
                continue

            token_pair = (min(token_1, token_2), max(token_1, token_2))

            if token_pair not in doc_seen_pairs:
                global_cooc_dict[token_pair] = global_cooc_dict.get(token_pair, 0.0) + 1.0
                doc_seen_pairs.add(token_pair)

            if token_1 not in doc_seen_words:
                global_word_dict[token_1] = global_word_dict.get(token_1, 0.0) + 1.0
                doc_seen_words.add(token_1)

            if token_2 not in doc_seen_words:
                global_word_dict[token_2] = global_word_dict.get(token_2, 0.0) + 1.0
                doc_seen_words.add(token_2)

    def __process_window_tf(global_cooc_dict, global_word_dict, token_ids, token_weights: List[float]):
        for j in range(1, len(token_ids)):
            value = min(token_weights[0], token_weights[j])

            token_1 = batch_dictionary[token_ids[0]]
            token_2 = batch_dictionary[token_ids[j]]

            if token_1 == token_2:
                continue

            if token_1 not in vocab or token_2 not in vocab:
                continue

            token_pair = (min(token_1, token_2), max(token_1, token_2))
            global_cooc_dict[token_pair] = global_cooc_dict.get(token_pair, 0.0) + value
            global_word_dict[token_1] = global_word_dict.get(token_1, 0.0) + 1.0
            global_word_dict[token_2] = global_word_dict.get(token_2, 0.0) + 1.0
            global_cooc_dict[RESERVED_TUPLE] += 2

    for item in batch.item:
        doc_seen_pairs = set()
        doc_seen_words = set()
        real_window_size = window_size if window_size > 0 else len(item.token_id)
        for window_start_id in range(len(item.token_id)):
            end_index = window_start_id + real_window_size
            token_ids = item.token_id[window_start_id: end_index if end_index < len(item.token_id) else len(item.token_id)]
            token_weights = item.token_weight[window_start_id: end_index if end_index < len(item.token_id) else len(item.token_id)]
            __process_window_df(global_cooc_df_dictionary, global_cooc_df_term_dictionary, token_ids, doc_seen_pairs, doc_seen_words)
            __process_window_tf(global_cooc_tf_dictionary, global_cooc_tf_term_dictionary, token_ids, token_weights)


def __size(global_cooc_dictionary):
    result = sys.getsizeof(global_cooc_dictionary) \
             + sum(sys.getsizeof(k) + sys.getsizeof(v) for k, v in global_cooc_dictionary.items())

    return result


def calculate_cooc(batches_path: str, vocab: List[str], window_size: int=10) -> CoocDictionaries:
    global_time_start = time.time()
    batches_list = glob.glob(os.path.join(batches_path, '*.batch'))

    logger.info(
        "Calculating cooc: %s batches were found in %s, start processing"
        % (len(batches_list), batches_path)
    )

    global_cooc_df_dictionary = dict()
    global_cooc_tf_dictionary = dict()
    global_cooc_df_term_dictionary = dict()
    global_cooc_tf_term_dictionary = dict()
    for index, filename in enumerate(batches_list):
        local_time_start = time.time()
        logger.debug('Processing batch: %s' % index)
        current_batch = artm.messages.Batch()
        with open(filename, 'rb') as fin:
            current_batch.ParseFromString(fin.read())
        __process_batch(
            global_cooc_df_dictionary, global_cooc_tf_dictionary,
            global_cooc_df_term_dictionary, global_cooc_tf_term_dictionary,
            current_batch, window_size,
            set(vocab)
        )

        logger.debug('Finished batch, elapsed time: %s' % (time.time() - local_time_start))

    logger.info(
        'Finished cooc dict collection, elapsed time: %s, size: %s Gb'
        % (time.time() - global_time_start, __size(global_cooc_tf_dictionary) / 1000000000.0)
    )

    return CoocDictionaries(
        cooc_df=global_cooc_df_dictionary,
        cooc_tf=global_cooc_tf_dictionary,
        cooc_df_term=global_cooc_df_term_dictionary,
        cooc_tf_term=global_cooc_df_term_dictionary
    )
