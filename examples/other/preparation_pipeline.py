import logging
import os.path

from autotm.preprocessing import PREPOCESSED_DATASET_FILENAME
from autotm.preprocessing.dictionaries_preparation import prepare_all_artifacts, prepearing_cooc_dict
from autotm.preprocessing.text_preprocessing import process_dataset

logger = logging.getLogger(__name__)


PATH_TO_DATASET = "../../data/sample_corpora/sample_dataset_lenta.csv"  # dataset with corpora to be processed
SAVE_PATH = "../../data/processed_sample_corpora"  # place where all the artifacts will be stored

col_to_process = "text"
dataset_name = "lenta_sample"
lang = "ru"  # available languages: ru, en
min_tokens_num = 3  # the minimal amount of tokens after processing to save the result


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
