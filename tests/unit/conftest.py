import os.path
from typing import List, Dict, Tuple

import pytest


def parse_vw(path: str) -> List[Tuple[str, Dict[str, float]]]:
    result = []

    with open(path, "r") as f:
        for line in f.readlines():
            elements = line.split(" ")
            word, tokens = elements[0], elements[1:]
            result.append((word, {token.split(':')[0]: float(token.split(':')[1]) for token in tokens}))

    return result\


@pytest.fixture(scope="session")
def test_corpora_path(pytestconfig: pytest.Config) -> str:
    return os.path.join(pytestconfig.rootpath, "data", "processed_lenta_ru_sample_corpora")
