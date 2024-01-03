import os.path
import tempfile

from autotm.preprocessing.cooc import calculate_cooc
from autotm.preprocessing.dictionaries_preparation import calculate_cooc_dicts, read_vocab, \
    convert_to_vw_format_and_save
from .conftest import parse_vw


def test_cooc(test_corpora_path: str):
    with tempfile.TemporaryDirectory() as tmpdir:
        # input data paths
        batches_path = os.path.join(test_corpora_path, "batches")
        vocab_path = os.path.join(test_corpora_path, "vocab.txt")

        # reading correct samples produced by old version of code
        cooc_df_vw = parse_vw(os.path.join(test_corpora_path, "cooc_df.txt"))
        cooc_tf_vw = parse_vw(os.path.join(test_corpora_path, "cooc_tf.txt"))
        ppmi_df_vw = parse_vw(os.path.join(test_corpora_path, "ppmi_df.txt"))
        ppmi_tf_vw = parse_vw(os.path.join(test_corpora_path, "ppmi_tf.txt"))
        vocab_words = read_vocab(vocab_path)

        # output paths
        cooc_file_path_df = os.path.join(tmpdir, "cooc_df.txt")
        cooc_file_path_tf = os.path.join(tmpdir, "cooc_tf.txt")

        # calculating
        cooc_dicts = calculate_cooc(batches_path=batches_path, window_size=5)

        convert_to_vw_format_and_save(cooc_dicts.cooc_df, vocab_words, cooc_file_path_df)
        convert_to_vw_format_and_save(cooc_dicts.cooc_tf, vocab_words, cooc_file_path_tf)

        # comparing
        produced_cooc_df_vw = parse_vw(cooc_file_path_df)
        produced_cooc_tf_vw = parse_vw(cooc_file_path_tf)

        assert produced_cooc_df_vw == cooc_df_vw
        assert produced_cooc_tf_vw == cooc_tf_vw
