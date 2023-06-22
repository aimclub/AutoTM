import os
import yaml
import json

import pandas as pd
import artm
from autotm.preprocessing.dictionaries_preparation import prepare_batch_vectorizer

PATH_TO_RUN_NAME = "tags/mlflow.runName"
PATH_TO_EXPERIMENT_ID = ""

PROCESSED_TEXT_COL = "processed_text"
TOP_TOPICS_COL = "SER_top_topics"


def get_experiment_path(
        exp_id: int, run_name: str, mlflow_path: str = "./mlruns/"
) -> str:
    """

    :param mlflow_path: path where mlflow stores the results default is
    :param exp_id: id of the experiment, which artifacts is needed
    :param run_name: name of the mlflow run
    :return: path to the experiment artifacts
    """
    for folder in os.listdir(mlflow_path):
        if folder != ".trash":
            subfolders = os.listdir(os.path.join(mlflow_path, folder))
            if len(subfolders) > 1:
                with open(os.path.join(mlflow_path, folder, "meta.yaml"), "r") as f:
                    data = yaml.load(f, Loader=yaml.Loader)
                if data["name"] == f"experiment_{exp_id}":
                    for subfolder in subfolders:
                        if subfolder != "meta.yaml":
                            with open(
                                    os.path.join(
                                        mlflow_path, folder, subfolder, PATH_TO_RUN_NAME
                                    )
                            ) as f:
                                f_run_name = f.read()
                                if f_run_name.strip() == run_name:
                                    return os.path.join(
                                        mlflow_path, folder, subfolder, "artifacts"
                                    )
    return None


def get_artifacts(artifacts_path: str):
    phi_folders = os.listdir(os.path.join(artifacts_path, "phi.csv"))
    theta_folders = os.listdir(os.path.join(artifacts_path, "theta.csv"))
    phi_matrix = pd.read_csv(os.path.join(artifacts_path, "phi.csv", phi_folders[0]))
    theta_matrix = pd.read_csv(
        os.path.join(artifacts_path, "theta.csv", theta_folders[0])
    )
    with open(os.path.join(artifacts_path, "topics.json")) as f:
        topics = json.load(f)
    return phi_matrix, theta_matrix, topics


# def _get_phi_dict_format(df):
#


def _transform_matrix(df: pd.DataFrame) -> pd.DataFrame:
    if "Unnamed: 0" in list(df):
        df = df.set_index("Unnamed: 0")
        df = df.T
    return df


def get_most_probable_topics_from_theta(
        df: pd.DataFrame, theta_df: pd.DataFrame, top_n: int = 2
) -> pd.DataFrame:
    """

    :param df: processed dataset which is produced by prepare_all function
    :param theta_df: theta matrix of trained model
    :param top_n: amount of top topics to consider
    :return: pd.Dataframe with 'top_topics' column
    """
    theta_df = _transform_matrix(theta_df)
    print(theta_df)
    assert (
            df.shape[0] == theta_df.shape[0]
    ), "Shapes of f and theta matrix are different"
    print(theta_df.apply(lambda x: ", ".join(x.nlargest(top_n).index.tolist()), axis=1))
    df[TOP_TOPICS_COL] = theta_df.apply(
        lambda x: ", ".join(x.nlargest(top_n).index.tolist()), axis=1
    ).tolist()
    return df


def get_top_words_from_topic_in_text(df: pd.DataFrame, topics: dict, top_w: int = 2):
    assert TOP_TOPICS_COL in list(
        df
    ), "use function get_most_probable_topics_from_theta to get top topics for texts"
    all_dicts = []
    for id, row in df.iterrows():
        topic_tokens = {}
        top_topics_names = [i.strip() for i in row[TOP_TOPICS_COL].split(",")]
        for topic in top_topics_names:
            words = set(row[PROCESSED_TEXT_COL].split()).intersection(
                set(topics[topic])
            )
            if len(words) > 1:
                topic_tokens[topic] = [word for word in topics[topic] if word in words][
                                      :top_w
                                      ]
        all_dicts.append(topic_tokens)
    df["SER_words_of_topic"] = all_dicts
    return df


def get_most_probable_words_from_phi(df, phi_df):
    raise NotImplementedError


class TopicsExtractor:
    def __init__(self, model_path):
        self.__tm_model_path = model_path
        self.model = self.__load_model()
        self.__tmp_batches_path = "./tmp/tmp_batches/"
        self.__tmp_dict_path = "./tmp/tmp_dict/"
        self.topics_dict = self.model.score_tracker["TopTokensScore"].last_tokens

    def info(self):
        pass

    def return_topics(self):
        pass

    def __load_model(self):
        return artm.load_artm_model(self.__tm_model_path)

    def get_prob_mixture(
            self,
            data_path,
            TMP_BATCHES_PATH="./tmp/tmp_batches/",
            TMP_DICT_PATH="./tmp/tmp_dict/",
            OUTPUT_DIR="./out",
            text_column_name="processed_text",
            input_format="csv",
            top_n=2,
    ):
        try:
            os.makedirs(TMP_BATCHES_PATH)
            os.makedirs(TMP_DICT_PATH)
        except:
            print("folders already exist")

        for tmp_batch in os.listdir(TMP_BATCHES_PATH):
            os.remove(os.path.join(TMP_BATCHES_PATH, tmp_batch))

        if input_format == "csv":
            posts = pd.read_csv(data_path)
        else:
            posts = pd.read_parquet(data_path)

        posts = posts[~posts[text_column_name].isna()]
        posts = posts.reset_index(drop=True)

        texts = posts[text_column_name].tolist()

        print("LEN {}".format(len(texts)))

        batch_vectorizer_test = prepare_batch_vectorizer(
            batches_dir=TMP_DICT_PATH,
            vw_path="./tmp/tmp_wv.txt",
            data_path=data_path,
            column_name=text_column_name,
        )

        theta_test = self.model.transform(batch_vectorizer=batch_vectorizer_test)
        theta_test_trans = theta_test.T
        main_topics = [i for i in list(theta_test_trans) if i.startswith("main")]
        theta_test_trans["top_topics"] = (
            theta_test_trans[main_topics]
            .apply(lambda x: ", ".join(x.nlargest(top_n).index.tolist()), axis=1)
            .tolist()
        )
        theta_test_trans = theta_test_trans.join(posts[[text_column_name]])
        theta_test_trans.to_csv(
            os.path.join(OUTPUT_DIR, "data_with_theta.csv"), index=None
        )

        print("All is saved to {} !".format(OUTPUT_DIR))
