import logging
import os
from typing import List, Union

import artm
import math
import pickle
import pandas as pd
import re

from autotm.preprocessing import PREPOCESSED_DATASET_FILENAME, RESERVED_TUPLE
from autotm.preprocessing.cooc import calculate_cooc
from autotm.utils import parallelize_dataframe
import itertools
from collections import Counter

logger = logging.getLogger(__name__)

# TODO: add inter-text coherence metrics (SemantiC, TopLen and FoCon)


def get_words_dict(text, stop_list):
    all_words = text
    words = sorted(set(all_words) - stop_list)
    return {w: all_words.count(w) for w in words}


def vocab_preparation(VOCAB_PATH, DICTIONARY_PATH):
    if not os.path.exists(VOCAB_PATH):
        with open(DICTIONARY_PATH, "r") as dictionary_file:
            with open(VOCAB_PATH, "w") as vocab_file:
                dictionary_file.readline()
                dictionary_file.readline()
                for line in dictionary_file:
                    elems = re.split(", ", line)
                    vocab_file.write(" ".join(elems[:2]) + "\n")


def _calculate_token_count():
    raise NotImplementedError


def _add_word_to_dict(word, w_dict):
    if word in w_dict:
        w_dict[word] += 1
    else:
        w_dict[word] = 1
    return w_dict


def _calculate_cooc_df_dict(data: list, vocab: List[str], window: int = 10) -> dict:
    cooc_df_dict = {}  # format dict{(tuple): cooc}
    term_freq_dict = {}
    existing_vocab = set(vocab)
    for text in data:
        document_cooc_df_dict = {}
        splitted = [word for word in text.split() if word in existing_vocab]
        for i in range(0, len(splitted) - window):
            for comb in itertools.combinations(splitted[i: i + window], 2):  # speed up
                comb = tuple(sorted(comb))  # adding comb sorting
                if comb in document_cooc_df_dict:
                    continue
                else:
                    document_cooc_df_dict[comb] = 1
                    term_freq_dict = _add_word_to_dict(comb[0], term_freq_dict)
                    term_freq_dict = _add_word_to_dict(comb[1], term_freq_dict)
        cooc_df_dict = dict(Counter(document_cooc_df_dict) + Counter(cooc_df_dict))
    return cooc_df_dict, term_freq_dict


def _calculate_cooc_tf_dict(data: list, vocab: List[str], window: int = 10) -> dict:
    term_freq_dict = {}
    cooc_tf_dict = {RESERVED_TUPLE: 0}  # format dict{(tuple): cooc}
    existing_vocab = set(vocab)
    for text in data:
        document_cooc_tf_dict = {}
        splitted = [word for word in text.split() if word in existing_vocab]
        for i in range(0, len(splitted) - window):
            for comb in itertools.combinations(splitted[i: i + window], 2):
                if comb in document_cooc_tf_dict:
                    document_cooc_tf_dict[comb] += 1
                else:
                    document_cooc_tf_dict[comb] = 1

                term_freq_dict = _add_word_to_dict(comb[0], term_freq_dict)
                term_freq_dict = _add_word_to_dict(comb[1], term_freq_dict)
                cooc_tf_dict[RESERVED_TUPLE] += 2
        # cooc_tf_dict = dict(Counter(document_cooc_tf_dict) + Counter(cooc_tf_dict))
        counter = Counter(document_cooc_tf_dict)
        counter.update(cooc_tf_dict)
        cooc_tf_dict = dict(counter)
    return cooc_tf_dict, term_freq_dict
    # local_num_of_pairs += 2
    # pass


def read_vocab(vocab_path: str) -> List[str]:
    # TODO: rewrite this part in case of several modalities
    vocab_words = []
    with open(vocab_path) as vpath:
        for line in vpath:
            splitted_line = line.split()
            if len(splitted_line) > 2:
                raise Exception("There are more than 2 modalities!")
            vocab_words.append(splitted_line[0].strip())

    return vocab_words


def calculate_ppmi(cooc_dict_path, n, term_freq_dict):
    print("Calculating pPMI...")
    ppmi_dict = {}
    with open(cooc_dict_path) as fopen:
        for line in fopen:
            splitted_line = line.split()
            ppmi_dict[splitted_line[0]] = [
                f'{word.split(":")[0].strip()}:{max(math.log2((float(word.split(":")[1]) / n) / (term_freq_dict[word.split(":")[0].strip()] / n * term_freq_dict[splitted_line[0]] / n)), 0)}'
                for word in splitted_line[1:]
            ]
    return ppmi_dict


# TODO: rewrite to storing in rb tree
def calculate_cooc_dicts(vocab: List[str], df: pd.DataFrame, window=10, n_cores=-1):
    """

    :param vocab: list of words to be used for calculating cooccurrences
    :param df: dataframe with 'processed_text'  column
    :param window: The size of window to collect cooccurrences in
    :param n_cores: available cores for parallel processing. Default: -1 (all)
    :return: cooc_df and cooc_tf dictionaries
    """
    data = df["processed_text"].tolist()
    cooc_df_dict = parallelize_dataframe(
        data,
        _calculate_cooc_df_dict,
        n_cores,
        return_type="dict",
        vocab=vocab,
        window=window,
    )
    cooc_tf_dict = parallelize_dataframe(
        data,
        _calculate_cooc_tf_dict,
        n_cores,
        return_type="dict",
        vocab=vocab,
        window=window,
    )
    return cooc_df_dict, cooc_tf_dict


def write_vw_dict(res_dict, vocab_words, fpath):
    with open(fpath, "w") as fopen:
        for word in vocab_words:
            try:
                fopen.write(f"{word}" + " " + " ".join(res_dict[word]) + "\n")
            except:
                # print(f'The word {word} is not found')
                pass
    print(f"{fpath} is ready!")


def convert_to_vw_format_and_save(cooc_dict, vocab_words, vw_path):
    if isinstance(cooc_dict, tuple):
        t_cooc_dict = cooc_dict[0]
    else:
        t_cooc_dict = cooc_dict
    data_dict = {}
    for item in sorted(t_cooc_dict.items(), key=lambda key: key[0]):
        if item == RESERVED_TUPLE:
            continue
        # word_1 = item[0][0]  # TODO: check this
        # word_2 = item[0][1]
        # if vocab_words.index(item[0][0]) > vocab_words.index(item[0][1]):
        #     word_2 = item[0][0]
        #     word_1 = item[0][1]
        if item[0][0] in data_dict:
            data_dict[item[0][0]].append(f"{item[0][1]}:{item[1]}")
        else:
            data_dict[item[0][0]] = [f"{item[0][1]}:{item[1]}"]
    write_vw_dict(data_dict, vocab_words, vw_path)


def prepearing_cooc_dict(
    BATCHES_DIR,
    WV_PATH,
    VOCAB_PATH,
    COOC_DICTIONARY_PATH,
    path_to_dataset,
    cooc_file_path_tf,
    cooc_file_path_df,
    ppmi_dict_tf,
    ppmi_dict_df,
    cooc_min_tf=0,
    cooc_min_df=0,
    cooc_window=10,
    n_cores=-1,
):
    """
    :param WV_PATH: path where to store data in Vowpal Wabbit format (https://github.com/VowpalWabbit/vowpal_wabbit/wiki/Input-format)
    :param VOCAB_PATH: path where the full dictionary is listed
    :param COOC_DICTIONARY_PATH: path to cooccurrence dictionary
    :param path_to_dataset: path to folder with dataset
    :param cooc_file_path_tf: path to tf coocurrances
    :param cooc_file_path_df: path to df coocurrances
    :param ppmi_dict_tf: path to ppmi tf dict
    :param ppmi_dict_df: path to ppmi df dict
    :param cooc_min_tf: Minimal number of documents to cooc in pairs to store the results
    :param cooc_min_df: Minimal number of documents to docs to cooc to store the results
    :param cooc_window: size of the window where to search for the cooccurrences
    :return:
    """

    vocab_words = read_vocab(VOCAB_PATH)

    data = pd.read_csv(path_to_dataset)
    docs_count = data.shape[0]

    logger.debug("Performing calculate_cooc_dicts")
    cooc_dicts = calculate_cooc(batches_path=BATCHES_DIR, vocab=vocab_words, window_size=cooc_window)
    logger.debug("Performed calculate_cooc_dicts")
    cooc_df_dict, cooc_df_term_dict = cooc_dicts.cooc_df, cooc_dicts.cooc_df_term
    cooc_tf_dict, cooc_tf_term_dict = cooc_dicts.cooc_tf, cooc_dicts.cooc_tf_term

    pairs_count = cooc_tf_dict[RESERVED_TUPLE]

    del cooc_tf_dict[RESERVED_TUPLE]

    convert_to_vw_format_and_save(cooc_df_dict, vocab_words, cooc_file_path_df)
    convert_to_vw_format_and_save(cooc_tf_dict, vocab_words, cooc_file_path_tf)

    logger.debug("Performing calculate_ppmi")
    ppmi_df = calculate_ppmi(cooc_file_path_df, docs_count, cooc_df_term_dict)
    ppmi_tf = calculate_ppmi(cooc_file_path_tf, pairs_count, cooc_tf_term_dict)
    logger.debug("Performed calculate_ppmi")

    write_vw_dict(ppmi_tf, vocab_words, ppmi_dict_tf)
    write_vw_dict(ppmi_df, vocab_words, ppmi_dict_df)

    # ! bigartm - c $WV_PATH - v $VOCAB_PATH - -cooc - window
    # 10 - -write - cooc - tf $cooc_file_path_tf - -write - cooc - df $cooc_file_path_df - -write - ppmi - tf $ppmi_dict_tf - -write - ppmi - df $ppmi_dict_df

    cooc_dict = artm.Dictionary()
    cooc_dict.gather(
        data_path=BATCHES_DIR,
        cooc_file_path=ppmi_dict_tf,
        vocab_file_path=VOCAB_PATH,
        symmetric_cooc_values=True,
    )
    cooc_dict.save_text(COOC_DICTIONARY_PATH)


def return_string_part(name_type, text):
    tokens = text.split()
    tokens = [item for item in tokens if item != ""]

    return " |" + name_type + " " + " ".join(["{}:1".format(token) for token in tokens])


def prepare_voc(batches_dir, vw_path, dataset: Union[pd.DataFrame, str], column_name="processed_text.txt"):
    print("Starting...")
    with open(vw_path, "w", encoding="utf8") as ofile:
        if isinstance(dataset, str):
            num_parts = 0
            try:
                for file in os.listdir(dataset):
                    if file.startswith("part"):
                        print("part_{}".format(num_parts), end="\r")
                        if file.split(".")[-1] == "csv":
                            part = pd.read_csv(os.path.join(dataset, file))
                        else:
                            part = pd.read_parquet(os.path.join(dataset, file))
                        part_processed = part[column_name].tolist()
                        for text in part_processed:
                            result = return_string_part("@default_class", text)
                            ofile.write(result + "\n")
                        num_parts += 1

            except NotADirectoryError:
                print("part 1/1")
                part = pd.read_csv(dataset)
                part_processed = part[column_name].tolist()
                for text in part_processed:
                    result = return_string_part("@default_class", text)
                    ofile.write(result + "\n")
        else:
            part_processed = dataset[column_name].tolist()
            for text in part_processed:
                result = return_string_part("@default_class", text)
                ofile.write(result + "\n")

    logger.info(" batches {} \n vocabulary {} \n are ready".format(batches_dir, vw_path))


def prepare_batch_vectorizer(
    batches_dir: str, vw_path: str, dataset: Union[pd.DataFrame, str], column_name: str = "processed_text"
):
    prepare_voc(batches_dir, vw_path, dataset, column_name=column_name)
    batch_vectorizer = artm.BatchVectorizer(
        data_path=vw_path,
        data_format="vowpal_wabbit",
        target_folder=batches_dir,
        batch_size=100,
    )

    return batch_vectorizer


def mutual_info_dict_preparation(fname):
    tokens_dict = {}

    with open(fname) as handle:
        for ix, line in enumerate(handle):
            list_of_words = line.strip().split()
            word_1 = list_of_words[0]
            for word_val in list_of_words[1:]:
                word_2, value = word_val.split(":")
                tokens_dict["{}_{}".format(word_1, word_2)] = float(value)
                tokens_dict["{}_{}".format(word_2, word_1)] = float(value)
    return tokens_dict


def prepare_all_artifacts(save_path: str):
    DATASET_PATH = os.path.join(save_path, PREPOCESSED_DATASET_FILENAME)
    BATCHES_DIR = os.path.join(save_path, "batches")
    WV_PATH = os.path.join(save_path, "test_set_data_voc.txt")
    COOC_DICTIONARY_PATH = os.path.join(save_path, "cooc_dictionary.txt")
    DICTIONARY_PATH = os.path.join(save_path, "dictionary.txt")
    VOCAB_PATH = os.path.join(save_path, "vocab.txt")
    cooc_file_path_df = os.path.join(save_path, "cooc_df.txt")
    cooc_file_path_tf = os.path.join(save_path, "cooc_tf.txt")
    ppmi_dict_df = os.path.join(save_path, "ppmi_df.txt")
    ppmi_dict_tf = os.path.join(save_path, "ppmi_tf.txt")
    MUTUAL_INFO_DICT_PATH = os.path.join(save_path, "mutual_info_dict.pkl")
    DOCUMENTS_TO_BATCH_PATH = os.path.join(save_path, PREPOCESSED_DATASET_FILENAME)

    # TODO: check why batch vectorizer is returned (unused further)
    logger.debug("Starting batch vectorizer...")
    prepare_batch_vectorizer(
        BATCHES_DIR, WV_PATH, DOCUMENTS_TO_BATCH_PATH
    )

    logger.debug("Preparing artm.Dictionary...")
    my_dictionary = artm.Dictionary()
    my_dictionary.gather(data_path=BATCHES_DIR, vocab_file_path=WV_PATH)
    my_dictionary.filter(min_df=3, class_id="text")
    my_dictionary.save_text(DICTIONARY_PATH)

    logger.debug("Vocabulary preparing...")
    vocab_preparation(VOCAB_PATH, DICTIONARY_PATH)

    logger.debug("Cooc dictionary preparing...")
    prepearing_cooc_dict(
        BATCHES_DIR,
        WV_PATH,
        VOCAB_PATH,
        COOC_DICTIONARY_PATH,
        DATASET_PATH,
        cooc_file_path_tf,
        cooc_file_path_df,
        ppmi_dict_tf,
        ppmi_dict_df,
    )

    logger.debug("Mutual info dictionary preparing...")
    mutual_info_dict = mutual_info_dict_preparation(ppmi_dict_tf)
    with open(MUTUAL_INFO_DICT_PATH, "wb") as handle:
        pickle.dump(mutual_info_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
