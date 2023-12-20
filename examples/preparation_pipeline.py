import logging

from autotm.preprocessing.dictionaries_preparation import prepare_all_artifacts
from autotm.preprocessing.text_preprocessing import process_dataset

logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s', level=logging.DEBUG, datefmt='%Y-%m-%d %H:%M:%S')
logger = logging.getLogger(__name__)


PATH_TO_DATASET = "../data/sample_corpora/sample_dataset_lenta.csv"  # dataset with corpora to be processed
SAVE_PATH = "../data/processed_sample_corpora"  # place where all the artifacts will be stored

col_to_process = "text"
dataset_name = "lenta_sample"
lang = "ru"  # available languages: ru, en
min_tokens_num = 3  # the minimal amount of tokens after processing to save the result

if __name__ == "__main__":
    logger.info("Stage 1: Dataset preparation")
    process_dataset(
        PATH_TO_DATASET,
        col_to_process,
        SAVE_PATH,
        lang,
        min_tokens_count=min_tokens_num,
    )

    logger.info("Stage 2: Prepare all artefacts")
    prepare_all_artifacts(SAVE_PATH)

    logger.info("All finished")
