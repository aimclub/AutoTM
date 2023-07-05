from autotm.preprocessing.dictionaries_preparation import (
    calculate_cooc_dicts, write_vw_dict,
    _add_word_to_dict)

import json
from contextlib import nullcontext as do_not_raise_exception
from json import JSONDecodeError
from getpass import getpass
from unittest.mock import patch, MagicMock

import pandas as pd

import pytest


DATASET_PROCESSED_TINY = pd.DataFrame(
    {
        "processed_text": [
            "this is text for testing purposes",
            "test the testing test to test the test",
            "the text is a good example of",
        ]
    }
)


@pytest.fixture()
def tiny_dataset_(tmpdir):
    dataset_processed = tmpdir.join("dataset.txt")
    dataset_processed.write(DATASET_PROCESSED_TINY)
    return dataset_processed


def test__add_word_to_dict():
    test_dict = {}
    assert _add_word_to_dict('test', test_dict) == {'test': 1}
