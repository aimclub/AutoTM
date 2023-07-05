import logging
import os
import pprint
import tempfile
from contextlib import contextmanager
from typing import Optional

import click
import pandas as pd
import yaml

from autotm.base import AutoTM

# TODO: add proper logging format initialization
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()


def obtain_autotm_params(
        config_path: str,
        topic_count: Optional[int],
        lang: Optional[str],
        alg: Optional[str],
        surrogate_alg: Optional[str],
        log_file: Optional[str]
):
    # TODO: define clearly format of config
    if config_path is not None:
        logger.info(f"Reading config from path: {os.path.abspath(config_path)}")
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
    else:
        config = dict()

    if topic_count is not None:
        config['topic_count'] = topic_count
    if lang is not None:
        pp = config.get('preprocessing_params', dict())
        pp['lang'] = lang
        config['preprocessing_params'] = pp
    if alg is not None:
        config['alg_name'] = alg
    if surrogate_alg is not None:
        config['surrogate_alg_name'] = surrogate_alg
    if log_file is not None:
        config['log_file_path'] = log_file

    return config


@contextmanager
def prepare_working_dir(working_dir: Optional[str] = None):
    if working_dir is None:
        with tempfile.TemporaryDirectory(prefix="autotm_wd_") as tmp_working_dir:
            yield tmp_working_dir
    else:
        yield working_dir


@click.group()
@click.option('-v', '--verbose', is_flag=True, show_default=True, default=False, help="Verbose output")
def cli(verbose: bool):
    if verbose:
        logging.basicConfig(level=logging.DEBUG, force=True)


@cli.command()
@click.option('--config', 'config_path', type=str, help="A path to config for fitting the model")
@click.option(
    '--working-dir',
    type=str,
    help="A path to working directory used by AutoTM for storing intermediate files. "
         "If not specified temporary directory will be created in the current directory "
         "and will be deleted upon successful finishing."
)
@click.option('--in', 'in_', type=str, required=True, help="A file in csv format with text corpus to build model on")
@click.option(
    '--out',
    type=str,
    default="mixtures.csv",
    help="A path to a file in csv format that will contain topic mixtures for texts from the incoming corpus"
)
@click.option('--model', type=str, default='model.artm', help="A path that will contain fitted ARTM model")
@click.option('-t', '--topic-count', type=int, help="Number of topics to fit model with")
@click.option('--lang', type=str, help='Language of the dataset')
@click.option('--alg', type=str, help="Hyperparameters tuning algorithm. Available: ga, bayes")
@click.option('--surrogate-alg', type=str, help="Surrogate algorithm to use.")
@click.option('--log-file', type=str, help="Log file path")
@click.option(
    '--overwrite',
    is_flag=True,
    show_default=True,
    default=False,
    help="Overwrite if model or/and mixture files already exist"
)
def fit(
        config_path: Optional[str],
        working_dir: Optional[str],
        in_: str,
        out: str,
        model: str,
        topic_count: Optional[int],
        lang: Optional[str],
        alg: Optional[str],
        surrogate_alg: Optional[str],
        log_file: Optional[str],
        overwrite: bool
):
    config = obtain_autotm_params(config_path, topic_count, lang, alg, surrogate_alg, log_file)

    logger.debug(f"Running AutoTM with params: {pprint.pformat(config, indent=4)}")

    logger.info(f"Loading data from {os.path.abspath(in_)}")
    df = pd.read_csv(in_)

    with prepare_working_dir(working_dir) as work_dir:
        logger.info(f"Using working directory {os.path.abspath(work_dir)} for AutoTM")
        autotm = AutoTM(
            **config,
            working_dir_path=work_dir
        )
        mixtures = autotm.fit_predict(df)

        logger.info(f"Saving model to {os.path.abspath(model)}")
        autotm.save(model, overwrite=overwrite)
        logger.info(f"Saving mixtures to {os.path.abspath(out)}")
        mixtures.to_csv(out, mode='w' if overwrite else 'x')

    logger.info("Finished AutoTM")


@cli.command()
@click.option('--model', type=str, required=True, help="A path to fitted saved ARTM model")
@click.option('--in', 'in_', type=str, required=True, help="A file in csv format with text corpus to build model on")
@click.option(
    '--out',
    type=str,
    default="mixtures.csv",
    help="A path to a file in csv format that will contain topic mixtures for texts from the incoming corpus"
)
@click.option(
    '--overwrite',
    is_flag=True,
    show_default=True,
    default=False,
    help="Overwrite if the mixture file already exists"
)
def predict(model: str, in_: str, out: str, overwrite: bool):
    logger.info(f"Loading model from {os.path.abspath(model)}")
    autotm_loaded = AutoTM.load(model)

    logger.info(f"Predicting mixtures for data from {os.path.abspath(in_)}")
    df = pd.read_csv(in_)
    mixtures = autotm_loaded.predict(df)

    logger.info(f"Saving mixtures to {os.path.abspath(out)}")
    mixtures.to_csv(out, mode='w' if overwrite else 'x')


if __name__ == "__main__":
    cli()
