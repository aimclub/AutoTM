import logging
import os
import time

from autotm.algorithms_for_tuning.genetic_algorithm.genetic_algorithm import run_algorithm
from autotm.infer import TopicsExtractor
from autotm.preprocessing.dictionaries_preparation import prepare_all_artifacts
from autotm.preprocessing.text_preprocessing import process_dataset


logger = logging.getLogger(__name__)


def test_fit_predict():
    # dataset with corpora to be processed
    path_to_dataset = "../data/sample_corpora/sample_dataset_lenta.csv"
    # place where all the artifacts will be stored
    save_processed_dataset_path = "../data/processed_sample_corpora"

    path_to_trained_model = "/home/nikolay/wspace/AutoTM/examples/mlruns/" \
                            "359967814084496879/1fbffcb5458646959a2ef92a455d7141/artifacts/model.artm/" \
                            "7f9c71e1-ae35-4b55-bb82-35630b2997da.artm"

    out_fpath = "./out"

    col_to_process = "text"
    dataset_name = "lenta_sample"
    lang = "ru"  # available languages: ru, en
    min_tokens_num = 3  # the minimal amount of tokens after processing to save the result
    num_iterations = 2
    topic_count = 10
    exp_id = int(time.time())

    logger.info(f"Experiment id: {exp_id}")

    use_nelder_mead_in_mutation = False
    use_nelder_mead_in_crossover = False
    use_nelder_mead_in_selector = False
    train_option = "offline"

    logger.info("Stage 1: Dataset preparation")
    process_dataset(
        path_to_dataset,
        col_to_process,
        save_processed_dataset_path,
        lang,
        min_tokens_count=min_tokens_num,
    )
    prepare_all_artifacts(save_processed_dataset_path)
    logger.info("Stage 2: Tuning the topic model")

    # exp_id and dataset_name will be needed further to store results in mlflow
    best_result = run_algorithm(
        data_path=save_processed_dataset_path,
        dataset=dataset_name,
        exp_id=exp_id,
        topic_count=topic_count,
        log_file="./log_file_test.txt",
        num_iterations=num_iterations,
        use_nelder_mead_in_mutation=use_nelder_mead_in_mutation,
        use_nelder_mead_in_crossover=use_nelder_mead_in_crossover,
        use_nelder_mead_in_selector=use_nelder_mead_in_selector,
        train_option=train_option,
    )

    # initializing the inference
    extractor = TopicsExtractor(path_to_trained_model)
    extractor.get_prob_mixture(
        data_path=os.path.join(save_processed_dataset_path, "ppp.csv"),
        OUTPUT_DIR=out_fpath,
    )

    # Note:
    # column 'top_topics' contains two most probable topics in the document
    # (see file: os.path.join(out_fpath, "data_with_theta.csv")
    # Note: it is also possible to cluster the embeddings to find interesting and meaningful combinations of topics

    # looking at topics
    for i in extractor.topics_dict:
        print()
        print(i)
        print(", ".join(extractor.topics_dict[i]))
