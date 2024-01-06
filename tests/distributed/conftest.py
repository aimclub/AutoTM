import docker
import pytest
import os

from docker.models.containers import Container
from docker.models.volumes import Volume


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


@pytest.fixture(scope='function')
def distributed_worker_setup(pytestconfig, shared_mlflow_runs_volume: Volume) -> Container:
    distributed_dataset_cache_path = os.path.join(pytestconfig.rootpath, 'tmp', 'distributed_dataset_cache')
    mlflow_runs_path = shared_mlflow_runs_volume.attrs['Options']['device']

    os.makedirs(distributed_dataset_cache_path, exist_ok=True)

    client = docker.from_env()

    # todo: clean existing containers

    # todo: create container with fitness worker
    # 1. use correctly built docker image
    # 2. mount mlflow runs folder
    # 3. mount shared folder for dataset artifacts
    fitness_worker_container = client.containers.create(image='', command='', )

    yield fitness_worker_container

    # todo: clean existing containers
