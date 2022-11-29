from autotm.preprocessing.dictionaries_preparation import calculate_cooc_dicts

import json
from contextlib import nullcontext as do_not_raise_exception
from json import JSONDecodeError
from getpass import getpass
from unittest.mock import patch, MagicMock

import pandas as pd

import pytest
import requests
from requests import exceptions

DATASET_TINY = pd.DataFrame({'processed_text': ['this is text for testing purposes',
                                                'test the testing test to test the test']})


@pytest.fixture()
def tiny_dataset_(tmpdir):
    dataset_fio = tmpdir.join("dataset.txt")
    dataset_fio.write(DATASET_TINY)
    return dataset_fio


def test_cooc_df_build():
    loaded_document = load_documents(filepath=tiny_dataset_fio)