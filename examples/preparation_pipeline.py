import logging
import os.path

from autotm.preprocessing import PREPOCESSED_DATASET_FILENAME
from autotm.preprocessing.dictionaries_preparation import prepare_all_artifacts, prepearing_cooc_dict
from autotm.preprocessing.text_preprocessing import process_dataset

logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s', level=logging.DEBUG, datefmt='%Y-%m-%d %H:%M:%S')
logger = logging.getLogger(__name__)


PATH_TO_DATASET = "../data/sample_corpora/sample_dataset_lenta.csv"  # dataset with corpora to be processed
SAVE_PATH = "../data/processed_sample_corpora"  # place where all the artifacts will be stored

# PATH_TO_DATASET = "../tmp/train-00000-of-00001.csv"  # dataset with corpora to be processed
# SAVE_PATH = "../tmp/train-00000-of-00001-processed-corpora"  # place where all the artifacts will be stored

col_to_process = "text"
dataset_name = "lenta_sample"
lang = "ru"  # available languages: ru, en
min_tokens_num = 3  # the minimal amount of tokens after processing to save the result

# def prepare_all_artifacts_debug(save_path: str):
#     DATASET_PATH = os.path.join(save_path, PREPOCESSED_DATASET_FILENAME)
#     BATCHES_DIR = os.path.join(save_path, "batches")
#     WV_PATH = os.path.join(save_path, "test_set_data_voc.txt")
#     COOC_DICTIONARY_PATH = os.path.join(save_path, "cooc_dictionary.txt")
#     DICTIONARY_PATH = os.path.join(save_path, "dictionary.txt")
#     VOCAB_PATH = os.path.join(save_path, "vocab.txt")
#     cooc_file_path_df = os.path.join(save_path, "cooc_df.txt")
#     cooc_file_path_tf = os.path.join(save_path, "cooc_tf.txt")
#     ppmi_dict_df = os.path.join(save_path, "ppmi_df.txt")
#     ppmi_dict_tf = os.path.join(save_path, "ppmi_tf.txt")
#     MUTUAL_INFO_DICT_PATH = os.path.join(save_path, "mutual_info_dict.pkl")
#     DOCUMENTS_TO_BATCH_PATH = os.path.join(save_path, PREPOCESSED_DATASET_FILENAME)
#
#     logger.debug("Cooc dictionary preparing...")
#     prepearing_cooc_dict(
#         BATCHES_DIR,
#         WV_PATH,
#         VOCAB_PATH,
#         COOC_DICTIONARY_PATH,
#         DATASET_PATH,
#         cooc_file_path_tf,
#         cooc_file_path_df,
#         ppmi_dict_tf,
#         ppmi_dict_df,
#     )
#
#
# if __name__ == "__main__":
#     prepare_all_artifacts_debug(SAVE_PATH)

# Run bigartm from cli
# bigartm \
#         -c vw \  # Raw corpus in Vowpal Wabbit format
#         -v vocab \  # vocab file in UCI format
#         --cooc-window 10 \
#         --cooc-min-tf 200 \
#         --write-cooc-tf cooc_tf_ \
#         --cooc-min-df 200 \
#         --write-cooc-df cooc_df_ \
#         --write-ppmi-tf ppmi_tf_ \
#         --write-ppmi-df ppmi_df_


# run container: docker run -it -v /home/nikolay/wspace/AutoTM/tmp/train-00000-of-00001-processed-corpora:/dataset artm:3.9 /bin/bash
# bigartm -c /dataset/test_set_data_voc.txt -v /dataset/vocab.txt --cooc-window 10 --cooc-min-tf 200 --write-cooc-tf cooc_tf_ --cooc-min-df 200 --write-cooc-df cooc_df_ --write-ppmi-tf ppmi_tf_ --write-ppmi-df ppmi_df_


# Normal version. DO NOT DELETE!!!
if __name__ == "__main__":
    logger.info("Stage 1: Dataset preparation")
    if not os.path.exists(SAVE_PATH):
        process_dataset(
            PATH_TO_DATASET,
            col_to_process,
            SAVE_PATH,
            lang,
            min_tokens_count=min_tokens_num,
        )
    else:
        logger.info("The preprocessed dataset already exists. Found files on path: %s" % SAVE_PATH)

    logger.info("Stage 2: Prepare all artefacts")
    prepare_all_artifacts(SAVE_PATH)

    logger.info("All finished")
