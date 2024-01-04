import itertools
import logging
import os.path
import tempfile
from typing import Dict

import artm
import pandas as pd

from autotm.preprocessing import PREPOCESSED_DATASET_FILENAME
from autotm.preprocessing.cooc import calculate_cooc
from autotm.preprocessing.dictionaries_preparation import read_vocab, \
    convert_to_vw_format_and_save, prepare_batch_vectorizer, vocab_preparation
from autotm.preprocessing.text_preprocessing import process_dataset
from .conftest import parse_vw

logger = logging.getLogger(__name__)

# first 5 docs lenta_ru sample, window_size == 10
CORRECT_COOC_TF = {
    'объект': {
        'площадь': 1.0,
        'участок': 1.0,
        'тысяча': 1.0,
        'квадрат': 1.0,
        'территория': 1.0,
        'бассейн': 1.0,
        'площадка': 1.0,
        'барбекю': 1.0,
        'продавец': 1.0,
        'выступать': 2.0,
        'трастовый': 1.0,
        'фонд': 1.0,
        'который': 1.0,
        'приобретать': 1.0,
        'назад': 1.0,
        'пора': 1.0,
        'особняк': 1.0,
        'перепродаваться': 1.0,
        'звезда': 5.0,
        'утверждение': 1.0,
        'ученый': 5.0,
        'первый': 1.0,
        'случай': 1.0,
        'удаваться': 1.0,
        'обнаруживать': 1.0,
        'обладать': 1.0,
        'кольцо': 3.0,
        'предел': 1.0,
        'солнечный': 1.0,
        'система': 1.0,
        'свой': 4.0,
        'результат': 2.0,
        'британский': 1.0,
        'астроном': 1.0,
        'докладывать': 1.0,
        'ежегодный': 1.0,
        'приводиться': 1.0,
        'сайт': 1.0,
        'университет': 1.0,
        'статья': 1.0,
        'исследователь': 2.0,
        'появляться': 1.0,
        'журнал': 1.0,
        'astrophysical': 1.0,
        'journal': 1.0,
        'исследование': 1.0,
        'swasp': 1.0,
        'располагать': 1.0,
        'расстояние': 1.0,
        'световой': 1.0,
        'земля': 1.0,
        'наблюдение': 2.0,
        'приходить': 1.0,
        'вывод': 1.0,
        'затмение': 1.0,
        'следствие': 2.0,
        'диск': 2.0,
        'проходить': 2.0,
        'вращаться': 2.0,
        'вокруг': 2.0,
        'точный': 2.0,
        'масса': 2.0,
        'определять': 2.0,
        'смочь': 2.0,
        'либо': 3.0,
        'коричневый': 1.0,
        'карлик': 1.0,
        'суперюпитер': 1.0,
        'место': 1.0,
        'получать': 1.0,
        'данные': 1.0,
        'проводиться': 1.0,
        'анализ': 1.0,
        'шутка': 1.0,
        'называть': 1.0,
        'сатурн': 1.0,
        'стероид': 1.0,
        'самый': 1.0,
        'большой': 1.0,
        'иметь': 1.0,
        'диаметр': 1.0,
        'десяток': 1.0,
        'миллион': 1.0,
        'километр': 1.0,
        'мнение': 1.0,
        'щель': 1.0,
        'мочь': 1.0,
        'работа': 1.0,
        'спутник': 1.0,
    },

    'astrophysical': {
        'изложение': 1.0,
        'доклад': 1.0,
        'приводиться': 1.0,
        'сайт': 1.0,
        'университет': 1.0,
        'статья': 1.0,
        'исследователь': 1.0,
        'появляться': 1.0,
        'журнал': 1.0,
        'journal': 1.0,
        'объект': 1.0,
        'исследование': 1.0,
        'ученый': 1.0,
        'выступать': 1.0,
        'звезда': 1.0,
        'swasp': 1.0,
        'располагать': 1.0,
        'расстояние': 1.0
    },
    'коричневый': {
        'астрофизик': 1.0,
        'университет': 1.0,
        'рочестер': 1.0,
        'обнаруживать': 1.0,
        'сатурн': 1.0,
        'стероид': 1.0,
        'карлик': 2.0,
        'газовый': 1.0,
        'гигант': 1.0,
        'который': 1.0,
        'вращаться': 2.0,
        'вокруг': 2.0,
        'молодой': 1.0,
        'звезда': 1.0,
        'утверждение': 1.0,
        'точный': 1.0,
        'масса': 1.0,
        'объект': 1.0,
        'ученый': 1.0,
        'определять': 1.0,
        'смочь': 1.0,
        'либо': 2.0,
        'суперюпитер': 1.0,
        'свой': 1.0,
        'очередь': 1.0,
        'необычный': 1.0,
        'кривая': 1.0,
        'блеск': 1.0,
        'объясняться': 1.0
    }
}

CORRECT_COOC_DF = {
    'объект': {
        'площадь': 1.0,
        'участок': 1.0,
        'тысяча': 1.0,
        'квадрат': 1.0,
        'территория': 1.0,
        'бассейн': 1.0,
        'площадка': 1.0,
        'барбекю': 1.0,
        'продавец': 1.0,
        'выступать': 1.0,
        'трастовый': 1.0,
        'фонд': 1.0,
        'который': 1.0,
        'приобретать': 1.0,
        'назад': 1.0,
        'пора': 1.0,
        'особняк': 1.0,
        'перепродаваться': 1.0,
        'звезда': 1.0,
        'утверждение': 1.0,
        'ученый': 1.0,
        'первый': 1.0,
        'случай': 1.0,
        'удаваться': 1.0,
        'обнаруживать': 1.0,
        'обладать': 1.0,
        'кольцо': 1.0,
        'предел': 1.0,
        'солнечный': 1.0,
        'система': 1.0,
        'свой': 1.0,
        'результат': 1.0,
        'британский': 1.0,
        'астроном': 1.0,
        'докладывать': 1.0,
        'ежегодный': 1.0,
        'приводиться': 1.0,
        'сайт': 1.0,
        'университет': 1.0,
        'статья': 1.0,
        'исследователь': 1.0,
        'появляться': 1.0,
        'журнал': 1.0,
        'astrophysical': 1.0,
        'journal': 1.0,
        'исследование': 1.0,
        'swasp': 1.0,
        'располагать': 1.0,
        'расстояние': 1.0,
        'световой': 1.0,
        'земля': 1.0,
        'наблюдение': 1.0,
        'приходить': 1.0,
        'вывод': 1.0,
        'затмение': 1.0,
        'следствие': 1.0,
        'диск': 1.0,
        'проходить': 1.0,
        'вращаться': 1.0,
        'вокруг': 1.0,
        'точный': 1.0,
        'масса': 1.0,
        'определять': 1.0,
        'смочь': 1.0,
        'либо': 1.0,
        'коричневый': 1.0,
        'карлик': 1.0,
        'суперюпитер': 1.0,
        'место': 1.0,
        'получать': 1.0,
        'данные': 1.0,
        'проводиться': 1.0,
        'анализ': 1.0,
        'шутка': 1.0,
        'называть': 1.0,
        'сатурн': 1.0,
        'стероид': 1.0,
        'самый': 1.0,
        'большой': 1.0,
        'иметь': 1.0,
        'диаметр': 1.0,
        'десяток': 1.0,
        'миллион': 1.0,
        'километр': 1.0,
        'мнение': 1.0,
        'щель': 1.0,
        'мочь': 1.0,
        'работа': 1.0,
        'спутник': 1.0,
    },

    'astrophysical': {
        'изложение': 1.0,
        'доклад': 1.0,
        'приводиться': 1.0,
        'сайт': 1.0,
        'университет': 1.0,
        'статья': 1.0,
        'исследователь': 1.0,
        'появляться': 1.0,
        'журнал': 1.0,
        'journal': 1.0,
        'объект': 1.0,
        'исследование': 1.0,
        'ученый': 1.0,
        'выступать': 1.0,
        'звезда': 1.0,
        'swasp': 1.0,
        'располагать': 1.0,
        'расстояние': 1.0
    },
    'коричневый': {
        'астрофизик': 1.0,
        'университет': 1.0,
        'рочестер': 1.0,
        'обнаруживать': 1.0,
        'сатурн': 1.0,
        'стероид': 1.0,
        'карлик': 1.0,
        'газовый': 1.0,
        'гигант': 1.0,
        'который': 1.0,
        'вращаться': 1.0,
        'вокруг': 1.0,
        'молодой': 1.0,
        'звезда': 1.0,
        'утверждение': 1.0,
        'точный': 1.0,
        'масса': 1.0,
        'объект': 1.0,
        'ученый': 1.0,
        'определять': 1.0,
        'смочь': 1.0,
        'либо': 1.0,
        'суперюпитер': 1.0,
        'свой': 1.0,
        'очередь': 1.0,
        'необычный': 1.0,
        'кривая': 1.0,
        'блеск': 1.0,
        'объясняться': 1.0
    }
}


def _standardize_cooc_dict(dictionary: Dict[str, Dict[str, float]]) -> Dict[str, Dict[str, float]]:
    pairs = ((min(w_1, w_2), max(w_1, w_2), value) for w_1, dct in dictionary.items() for w_2, value in dct.items())
    pairs = sorted(pairs, key=lambda x: x[0])
    gpairs = itertools.groupby(pairs, key=lambda x: x[0])
    return {w_1: {w_2: value for _, w_2, value in group} for w_1, group in gpairs}


def test_cooc(pytestconfig):
    col_to_process = 'text'
    dataset_path = os.path.join(pytestconfig.rootpath, "../data/sample_corpora/sample_dataset_lenta.csv")
    df = pd.read_csv(dataset_path)
    df = df.iloc[:5]

    with tempfile.TemporaryDirectory() as tmpdir:
        save_path = os.path.join(tmpdir, "processed_sample_corpora")
        batches_path = os.path.join(tmpdir, "batches")
        wv_path = os.path.join(tmpdir, "test_set_data_voc.txt")
        dictionary_path = os.path.join(tmpdir, "dictionary.txt")
        vocab_path = os.path.join(tmpdir, "vocab.txt")
        documents_to_batch_path = os.path.join(save_path, PREPOCESSED_DATASET_FILENAME)

        process_dataset(
            df,
            col_to_process=col_to_process,
            save_path=save_path,
            min_tokens_count=0
        )

        logger.debug("Starting batch vectorizer...")
        prepare_batch_vectorizer(
            batches_path, wv_path, documents_to_batch_path
        )

        logger.debug("Preparing artm.Dictionary...")
        my_dictionary = artm.Dictionary()
        my_dictionary.gather(data_path=batches_path, vocab_file_path=wv_path)
        my_dictionary.save_text(dictionary_path)

        logger.debug("Vocabulary preparing...")
        vocab_preparation(vocab_path, dictionary_path)

        vocab_words = read_vocab(vocab_path)

        # output paths
        cooc_file_path_df = os.path.join(tmpdir, "cooc_df.txt")
        cooc_file_path_tf = os.path.join(tmpdir, "cooc_tf.txt")

        # calculating
        cooc_dicts = calculate_cooc(batches_path=batches_path, vocab=vocab_words, window_size=10)

        convert_to_vw_format_and_save(cooc_dicts.cooc_df, vocab_words, cooc_file_path_df)
        convert_to_vw_format_and_save(cooc_dicts.cooc_tf, vocab_words, cooc_file_path_tf)

        # comparing
        produced_cooc_df_vw = parse_vw(cooc_file_path_df)
        produced_cooc_tf_vw = parse_vw(cooc_file_path_tf)

        reduced_cooc_df_vw = {word: produced_cooc_df_vw[word] for word in CORRECT_COOC_DF}
        reduced_cooc_tf_vw = {word: produced_cooc_tf_vw[word] for word in CORRECT_COOC_TF}

        assert reduced_cooc_df_vw == CORRECT_COOC_DF
        assert reduced_cooc_tf_vw == CORRECT_COOC_TF
