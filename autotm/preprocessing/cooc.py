from __future__ import print_function

import codecs
import glob
import logging
import os
import sys
import tempfile
import time
from dataclasses import dataclass
from typing import Dict, Tuple

import artm
from six import iteritems
from six.moves import range


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


def __save_dictionary(cooc_dictionary, num_tokens):
    with open('cooc_data.txt', 'w') as fout:
        for index in range(num_tokens):
            if index in cooc_dictionary:
                for key, value in iteritems(cooc_dictionary[index]):
                    fout.write(u'{0} {1} {2}\n'.format(index, key, value))


def __process_batch(global_cooc_dictionary, batch, window_size, dictionary):
    batch_dictionary = __create_batch_dictionary(batch)

    def __process_window(token_ids, token_weights):
        for j in range(1, len(token_ids)):
            value = min(token_weights[0], token_weights[j])
            token_index_1 = dictionary[batch_dictionary[token_ids[0]]]
            token_index_2 = dictionary[batch_dictionary[token_ids[j]]]

            if token_index_1 in global_cooc_dictionary:
                if token_index_2 in global_cooc_dictionary:
                    if token_index_2 in global_cooc_dictionary[token_index_1]:
                        global_cooc_dictionary[token_index_1][token_index_2] += value
                    else:
                        if token_index_1 in global_cooc_dictionary[token_index_2]:
                            global_cooc_dictionary[token_index_2][token_index_1] += value
                        else:
                            global_cooc_dictionary[token_index_1][token_index_2] = value
                else:
                    if token_index_2 in global_cooc_dictionary[token_index_1]:
                        global_cooc_dictionary[token_index_1][token_index_2] += value
                    else:
                        global_cooc_dictionary[token_index_1][token_index_2] = value
            else:
                if token_index_2 in global_cooc_dictionary:
                    if token_index_1 in global_cooc_dictionary[token_index_2]:
                        global_cooc_dictionary[token_index_2][token_index_1] += value
                    else:
                        global_cooc_dictionary[token_index_2][token_index_1] = value
                else:
                    global_cooc_dictionary[token_index_1] = {}
                    global_cooc_dictionary[token_index_1][token_index_2] = value

    for item in batch.item:
        real_window_size = window_size if window_size > 0 else len(item.token_id)
        for window_start_id in range(len(item.token_id)):
            end_index = window_start_id + real_window_size
            token_ids = item.token_id[window_start_id: end_index if end_index < len(item.token_id) else len(item.token_id)]
            token_weights = item.token_weight[window_start_id: end_index if end_index < len(item.token_id) else len(item.token_id)]
            __process_window(token_ids, token_weights)

def __size(global_cooc_dictionary):
    result = sys.getsizeof(global_cooc_dictionary)
    for k_1, internal in iteritems(global_cooc_dictionary):
        result += sys.getsizeof(k_1)
        for t, v in iteritems(internal):
            result += sys.getsizeof(t)
            result += sys.getsizeof(v)

    return result


def calculate_cooc(batches_path: str, vocab_path: str, window_size: int=10) -> CoocDictionaries:
    encoding = 'utf-8'
    global_time_start = time.time()
    batches_list = glob.glob(os.path.join(batches_path, '*.batch'))

    logger.info(
        "Calculating cooc: %s batches were found in %s, start processing using vocab from %s"
        % (len(batches_list), batches_path, vocab_path)
    )

    with tempfile.NamedTemporaryFile(mode='w+', delete=False) as fp:
        fp.close()

        temp_dict = artm.Dictionary()
        temp_dict.load_text(vocab_path, encoding)
        temp_dict.save_text(fp.name)

        dictionary = {}
        with codecs.open(fp.name, 'r', encoding) as fin:
            next(fin)
            next(fin)
            for index, line in enumerate(fin):
                dictionary[line.split(' ')[0][0: -1]] = index

    # tf dict
    global_cooc_dictionary = {}
    for index, filename in enumerate(batches_list):
        local_time_start = time.time()
        logger.debug('Processing batch: %s' % index)
        current_batch = artm.messages.Batch()
        with open(filename, 'rb') as fin:
            current_batch.ParseFromString(fin.read())
        __process_batch(global_cooc_dictionary, current_batch, window_size, dictionary)

        logger.debug('Finished batch, elapsed time: %s' % (time.time() - local_time_start))

    logger.info(
        'Finished cooc dict collection, elapsed time: %s, size: %s Gb'
        % (time.time() - global_time_start, __size(global_cooc_dictionary) / 1000000000.0)
    )

    return cooc_df_dict, global_cooc_dictionary
