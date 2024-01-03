import os.path

from .conftest import parse_vw


def test_cooc(test_corpora_path: str):
    # reading correct samples produced by old version of code
    cooc_df_vw = parse_vw(os.path.join(test_corpora_path, "cooc_df.txt"))
    cooc_tf_vw = parse_vw(os.path.join(test_corpora_path, "cooc_tf.txt"))
    ppmi_df_vw = parse_vw(os.path.join(test_corpora_path, "ppmi_df.txt"))
    ppmi_tf_vw = parse_vw(os.path.join(test_corpora_path, "ppmi_tf.txt"))
