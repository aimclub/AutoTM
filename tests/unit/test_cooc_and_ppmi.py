import logging
import os.path
import tempfile

import artm

from autotm.preprocessing import PREPOCESSED_DATASET_FILENAME
from autotm.preprocessing.cooc import calculate_cooc
from autotm.preprocessing.dictionaries_preparation import read_vocab, \
    convert_to_vw_format_and_save, prepare_batch_vectorizer, vocab_preparation
from autotm.preprocessing.text_preprocessing import process_dataset
from .conftest import parse_vw

logger = logging.getLogger(__name__)


def test_cooc(pytestconfig):
    dataset_path = os.path.join(pytestconfig.rootpath, "../data/sample_corpora/sample_dataset_lenta.csv")

    with tempfile.TemporaryDirectory() as tmpdir:
        save_path = os.path.join(tmpdir, "processed_sample_corpora")
        batches_path = os.path.join(tmpdir, "batches")
        wv_path = os.path.join(tmpdir, "test_set_data_voc.txt")
        dictionary_path = os.path.join(tmpdir, "dictionary.txt")
        vocab_path = os.path.join(tmpdir, "vocab.txt")
        documents_to_batch_path = os.path.join(save_path, PREPOCESSED_DATASET_FILENAME)

        process_dataset(
            dataset_path,
            col_to_process='text',
            save_path=save_path,
            lang='ru',
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

        # reading correct samples produced by old version of code
        # cooc_df_vw = parse_vw(os.path.join(tmpdir, "cooc_df.txt"))
        # cooc_tf_vw = parse_vw(os.path.join(tmpdir, "cooc_tf.txt"))
        # ppmi_df_vw = parse_vw(os.path.join(tmpdir, "ppmi_df.txt"))
        # ppmi_tf_vw = parse_vw(os.path.join(tmpdir, "ppmi_tf.txt"))
        vocab_words = read_vocab(vocab_path)

        # output paths
        cooc_file_path_df = os.path.join(tmpdir, "cooc_df.txt")
        cooc_file_path_tf = os.path.join(tmpdir, "cooc_tf.txt")

        # calculating
        cooc_dicts = calculate_cooc(batches_path=batches_path, vocab=vocab_words, window_size=20)

        convert_to_vw_format_and_save(cooc_dicts.cooc_df, vocab_words, cooc_file_path_df)
        convert_to_vw_format_and_save(cooc_dicts.cooc_tf, vocab_words, cooc_file_path_tf)

        # comparing
        produced_cooc_df_vw = parse_vw(cooc_file_path_df)
        produced_cooc_tf_vw = parse_vw(cooc_file_path_tf)

        k = 0

        # assert produced_cooc_df_vw == cooc_df_vw
        # assert produced_cooc_tf_vw == cooc_tf_vw
