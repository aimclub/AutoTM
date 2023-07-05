import logging
import os
import shutil
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from typing import Optional, Tuple, List, cast, ContextManager, Dict

import artm
import gridfs
import mlflow
from pymongo import MongoClient

from autotm.fitness.tm import TopicModel
from autotm.schemas import IndividualDTO
from autotm.utils import TimeMeasurements

logger = logging.getLogger()


@dataclass
class TopicModelFiles:
    model_dir: str
    model_archive: str
    phi: str
    theta: str

    @property
    def files(self) -> Dict[str, str]:
        """
        :return: Dict of (name,filepath) pairs
        """
        return {"model": self.model_archive, "phi": self.phi, "theta": self.theta}


@contextmanager
def model_files(tm: TopicModel) -> ContextManager[TopicModelFiles]:
    uid = tm.uid
    base_path = f"/tmp/tm-{uid}"

    tmp_file = os.path.join(base_path, f"{uid}.artm")
    tmp_zip_file = os.path.join(base_path, f"{uid}.artm.zip")
    tmp_theta_file = os.path.join(base_path, f"theta-{uid}.csv")
    tmp_phi_file = os.path.join(base_path, f"phi-{uid}.csv")

    os.makedirs(base_path)

    tm.save_model(path=tmp_file)

    tmp_zip_file_base_name, _ = os.path.splitext(tmp_zip_file)
    shutil.make_archive(tmp_zip_file_base_name, "zip", tmp_file)

    model = cast(artm.ARTM, tm.model)
    theta_matrix = model.get_theta()
    phi_matrix = model.get_phi()

    theta_matrix.to_csv(tmp_theta_file, header=True)
    phi_matrix.to_csv(tmp_phi_file, header=True)

    yield TopicModelFiles(
        model_dir=tmp_file,
        model_archive=tmp_zip_file,
        phi=tmp_phi_file,
        theta=tmp_theta_file,
    )

    shutil.rmtree(base_path, ignore_errors=True)


def make_readable_topics(tm: TopicModel) -> str:
    topics = tm.get_topics()

    def topic_seq_num(pair: Tuple[str, List[str]]) -> int:
        tname, _ = pair
        if tname.startswith("main"):
            return int(tname[len("main"):])
        if tname.startswith("back"):
            return tm.topic_count + int(tname[len("back"):])
        return -1

    ordered_topics = sorted(topics.items(), key=topic_seq_num)
    topics_reprs = [
        f"{topic}:{os.linesep}{os.linesep.join(' '.join(words[i:i + 10]) for i in range(0, len(words), 10))}"
        for topic, words in ordered_topics
    ]
    full_repr = "\n\n".join(topics_reprs)
    return full_repr


def log_params_and_artifacts(
    tm: TopicModel,
    tm_files: TopicModelFiles,
    individual: IndividualDTO,
    time_metrics: TimeMeasurements,
    alg_args: Optional[str],
    is_tmp: bool = False,
):
    logger.info("Logging params and artifacts to mlflow")
    logger.info(f"Created experiment_{individual.exp_id}")
    run_name = f"fitness-{individual.dataset}-{uuid.uuid4()}"
    if is_tmp:
        run_name += f"_tmp_{individual.iteration_id}"

    try:
        experiment_id = mlflow.create_experiment(f"experiment_{individual.exp_id}")
    except:
        experiment = mlflow.get_experiment_by_name(f"experiment_{individual.exp_id}")
        experiment_id = experiment.experiment_id
        logger.info("Experiment exists, omitting creation")

    # mlflow.delete_experiment

    print(f"Experiment run name: {run_name}")
    # try:
    with mlflow.start_run(run_name=run_name, experiment_id=experiment_id):
        params = {
            "uid": tm.uid,
            "dataset": individual.dataset,
            "fitness_name": individual.fitness_name,
            "exp_id": individual.exp_id,
        }
        d = individual.make_params_dict()
        logger.debug(f"Params dict: {d}")
        params.update(d)

        artifact_path = "model.artm"
        theta_artifact_path = "theta.csv"
        phi_artifact_path = "phi.csv"
        individual_artifact_path = "individual.json"
        metrics_artifact_path = "metrics.json"
        time_metrics_artifact_path = "time_metrics.json"
        topics_artifact_path = "topics.json"
        readable_topics_artifact_path = "topics.txt"
        alg_args_artifact_path = "alg_args.txt"

        topics = tm.get_topics()
        full_repr_topics = make_readable_topics(tm)

        mlflow.log_params(params)
        mlflow.log_dict(individual.dict(), artifact_file=individual_artifact_path)
        mlflow.log_dict(individual.fitness_value, artifact_file=metrics_artifact_path)
        mlflow.log_dict(time_metrics, artifact_file=time_metrics_artifact_path)
        mlflow.log_artifact(local_path=tm_files.model_dir, artifact_path=artifact_path)
        mlflow.log_artifact(
            local_path=tm_files.theta, artifact_path=theta_artifact_path
        )
        mlflow.log_artifact(local_path=tm_files.phi, artifact_path=phi_artifact_path)
        mlflow.log_dict(topics, artifact_file=topics_artifact_path)
        mlflow.log_text(
            alg_args if alg_args else "", artifact_file=alg_args_artifact_path
        )
        mlflow.log_text(full_repr_topics, artifact_file=readable_topics_artifact_path)
    # except:
    #     logger.info("The model is already logged")

    logger.info("Logged params and artifacts to mlflow")


def log_stats(
    tm: TopicModel,
    tm_files: TopicModelFiles,
    individual: IndividualDTO,
    time_metrics: TimeMeasurements,
    alg_args: Optional[str],
):
    logger.info("Logging run stats to mongodb")

    uid = tm.uid

    if "MONGO_URI" not in os.environ:
        raise Exception("Unable to find mongo uri - MONGO_URI env variable not found")

    mongo_collection = os.environ.get("MONGO_COLLECTION", "tm_stats")

    with MongoClient(os.environ["MONGO_URI"]) as client:
        db = client.tm_experiments_runs
        logger.info("Writing main model's files to mongo GridFS")
        fs = gridfs.GridFS(db)

        gridfs_files = dict()
        for name, file_path in tm_files.files.items():
            with open(file_path, "rb") as f:
                file_id = fs.put(f, filename=f"{name}-{uid}.zip", parent_uid=f"{uid}")
            gridfs_files[f"{name}_file_id"] = file_id

        logger.info("Writing main stats")
        dt = datetime.now()
        stats = {
            "uid": str(uid),
            "datetime": dt.strftime("%Y-%m-%d %H:%M:%S.%f"),
            "timestamp": dt.timestamp(),
            "individual": individual.dict(),
            "alg_args": alg_args,
            "params": individual.make_params_dict(),
            "topics": tm.get_topics(),
            "time_metrics": time_metrics,
            **gridfs_files,
        }
        collection = db[mongo_collection]
        uid = collection.insert_one(stats).inserted_id

    logger.info(f"Inserted into mongodb stats, received uid {uid}")


@contextmanager
def succeed_or_log_error():
    try:
        yield
    except Exception as ex:
        logger.error(
            "Exception occured while trying to execute the action", exc_info=ex
        )
