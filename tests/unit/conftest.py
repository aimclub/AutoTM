import itertools
import os.path
from typing import Dict

import pytest

from autotm.fitness.tm import ENV_AUTOTM_LLM_API_KEY


def parse_vw(path: str) -> Dict[str, Dict[str, float]]:
    result = dict()

    with open(path, "r") as f:
        for line in f.readlines():
            elements = line.split(" ")
            word, tokens = elements[0], elements[1:]
            if word in result:
                raise ValueError("The word is repeated")
            result[word] = {token.split(':')[0]: float(token.split(':')[1]) for token in tokens}

    pairs = ((min(word_1, word_2), max(word_1, word_2), value) for word_1, pairs in result.items() for word_2, value in pairs.items())
    pairs = sorted(pairs, key=lambda x: x[0])
    gpairs = itertools.groupby(pairs, key=lambda x: x[0])
    ordered_result = {word_1: {word_2: value for _, word_2, value in pps} for word_1, pps in gpairs}
    return ordered_result


@pytest.fixture(scope="session")
def test_corpora_path(pytestconfig: pytest.Config) -> str:
    return os.path.join(pytestconfig.rootpath, "data", "processed_lenta_ru_sample_corpora")

@pytest.fixture(scope="session")
def openai_api_key() -> str:
    if ENV_AUTOTM_LLM_API_KEY not in os.environ:
        raise ValueError(f"Env var {ENV_AUTOTM_LLM_API_KEY} with openai API key is not set")
    return os.environ[ENV_AUTOTM_LLM_API_KEY]
