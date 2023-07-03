import os

from autotm.infer import TopicsExtractor
from autotm.preprocessing.text_preprocessing import process_dataset

# path_to_trained_model = (
#     "path_to_trained_model"  # path to folder with n_wt.bin, score_tracker.bin, etc
# )

path_to_trained_model = "/home/nikolay/wspace/AutoTM/examples/mlruns/" \
                        "359967814084496879/1fbffcb5458646959a2ef92a455d7141/artifacts/model.artm/" \
                        "7f9c71e1-ae35-4b55-bb82-35630b2997da.artm"

path_to_dataset = "../data/sample_corpora/sample_dataset_lenta.csv"
save_processed_dataset_path = "../data/processed_sample_corpora/"
col_to_process = "text"
lang = "ru"  # available languages: ru, en
min_tokens_num = 3  # the minimal amount of tokens after processing to save the result
out_fpath = "./out"
os.mkdir(out_fpath)


def main():
    # preprocessing data
    process_dataset(
        path_to_dataset,
        col_to_process,
        save_processed_dataset_path,
        lang,
        min_tokens_count=min_tokens_num,
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


if __name__ == "__main__":
    main()
