import pytest
import os


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


@pytest.fixture(scope="session")
def docker_compose_file(pytestconfig):
    return os.path.join(str(pytestconfig.rootpath), "tests", "deploy", "compose.yaml")
