import logging
import shutil
import subprocess
from typing import cast, Dict

import docker
import pytest
import os

from docker.models.containers import Container
from docker.models.volumes import Volume


logger = logging.getLogger(__name__)


################## Docker compose fixtures ##################
@pytest.fixture(scope="session")
def docker_compose_file(pytestconfig):
    return os.path.join(str(pytestconfig.rootpath), "tests", "deploy", "compose.yaml")

# Override the following 3 fixtures to make pytest-docker running with externally started docker compose
@pytest.fixture(scope="session")
def docker_compose_project_name():
    return ""


@pytest.fixture(scope="session")
def docker_setup():
    return None


@pytest.fixture(scope="session")
def docker_cleanup():
    return None
#########################################

@pytest.fixture(scope="session")
def shared_mlflow_runs_volume(pytestconfig, docker_setup) -> Volume:
    mlflow_volume_id = 'test-autotm-mlflow-runs-data'
    client = docker.from_env()
    vs = [v for v in client.volumes.list() if v.id == mlflow_volume_id]
    if len(vs) == 0:
        source_path = os.path.join(pytestconfig.rootpath, 'tmp', 'mlflow-shared-mlruns')
        os.makedirs(source_path, exist_ok=True)
        mlflow_volume = client.volumes.create(
            name=mlflow_volume_id, driver='local',
            driver_opts={'o': 'bind', 'device': source_path}
        )
    elif len(vs) > 1:
        raise ValueError(f"Found several volumes with id: {mlflow_volume_id}")
    else:
        mlflow_volume = vs[0]

    yield mlflow_volume


@pytest.fixture(scope="session")
def fitness_worker_image(pytestconfig, wms_installation, k8s_testing_config) -> str:
    image_name = "fitness-worker:test-autotm"
    dockerfile_path = os.path.join(pytestconfig.rootpath, 'cluster', 'docker', 'worker.dockerfile')

    logger.info(f"Building image {image_name}")

    proc = subprocess.run(
        f"poetry build "
        f"&& poetry export --without-hashes > requirements.txt"
        f"&& docker build -t {image_name} -f {dockerfile_path} .",
        shell=True,
        cwd=pytestconfig.rootpath
    )

    if proc.returncode != 0:
        raise RuntimeError(f"Failed to build interactive worker image {image_name}")

    logger.info(f"Image {image_name} has been built")

    return image_name


@pytest.fixture(scope='function')
def distributed_worker_setup(pytestconfig, shared_mlflow_runs_volume: Volume, fitness_worker_image: str) \
        -> Dict[str, str]:
    labels = {'autotm': 'fitness_worker'}
    filter_label = ','.join([f'{k}={v}' for k, v in labels.items()])
    distributed_dataset_cache_path = os.path.join(pytestconfig.rootpath, 'tmp', 'distributed_dataset_cache')
    # to find a path the volume points to
    # mlflow_runs_path = shared_mlflow_runs_volume.attrs['Options']['device']

    client = docker.from_env()

    def clean_env():
        containers = client.containers.list(all=True, filters={'label': [filter_label]})
        for cnt in containers:
            cnt = cast(Container, cnt)
            cnt.stop()
            cnt.remove()

        if os.path.exists(distributed_dataset_cache_path):
            shutil.rmtree(distributed_dataset_cache_path)

    # prepare folders
    os.makedirs(distributed_dataset_cache_path)

    # remove old containers if exist
    clean_env()

    # create fitness worker
    fitness_worker_container = client.containers.run(
        detach=True,
        network="deploy_test_autotm_net",
        ports={"5000": "5000"},
        image=fitness_worker_image,
        hostname="fitness_worker",
        environment={
            "CELERY_BROKER_URL": "amqp://guest:guest@rabbitmq:5672",
            "CELERY_RESULT_BACKEND": "redis://redis:6379/1",
            "NUM_PROCESSORS": "4",
            "AUTOTM_COMPONENT": "worker",
            "AUTOTM_EXEC_MODE": "cluster",
            "DATASETS_CONFIG": "/etc/fitness/datasets-config.yaml",
            "MLFLOW_TRACKING_URI": "mysql+pymysql://mlflow:mlflow@mlflow-db:3306/mlflow",
            "MONGO_URI": "mongodb://mongoadmin:secret@monngo:27017",
            "MONGO_COLLECTION": "tm_stats"
        },
        volumes=[
            f"{shared_mlflow_runs_volume.id}:/var/lib/mlruns",
            f"{distributed_dataset_cache_path}:/distributed_dataset_cache"
        ],
        labels=labels
    )

    fitness_computing_settings = {
        'AUTOTM_COMPONENT': 'head',
        'AUTOTM_EXEC_MODE': 'cluster',
        'CELERY_BROKER_URL': 'amqp://guest:guest@localhost:5672',
        'CELERY_RESULT_BACKEND': 'redis://localhost:6379/1'
    }

    for env_var, value in fitness_computing_settings:
        os.environ[env_var] = value

    yield fitness_computing_settings

    # remove containers if exist
    clean_env()
