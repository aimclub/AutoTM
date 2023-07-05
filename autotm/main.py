import logging
import tempfile
from contextlib import contextmanager
from typing import Optional

import click
import yaml

from autotm.base import AutoTM
import pandas as pd


# TODO: add proper logging format initialization
logger = logging.getLogger()


@contextmanager
def prepare_working_dir(working_dir: Optional[str] = None):
    if working_dir is None:
        with tempfile.TemporaryDirectory(prefix="autotm_wd_") as tmp_working_dir:
            yield tmp_working_dir
    else:
        yield working_dir


# TODO: add logging level
@click.group()
def cli():
    pass


# TODO: add lang and log_file params
@cli.command()
@click.option('--config', type=str, help="A path to config for fitting the model")
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
@click.option('--alg', type=str, help="Hyperparameters tuning algorithm. Available: ga, bayes")
@click.option('--surrogate-alg', type=str, help="Surrogate algorithm to use.")
def fit(
        config: Optional[str],
        working_dir: Optional[str],
        in_: str,
        out: str,
        model: str,
        t: Optional[int],
        alg: Optional[str],
        surrogate_alg: Optional[str]
):
    # TODO: define clearly format of config
    if config is not None:
        with open(config, "r") as f:
            config = yaml.load(f)
    else:
        config = dict()

    cli_params = dict()
    if t is not None:
        cli_params['topic_count'] = t
    if alg is not None:
        cli_params['alg_name'] = alg
    if surrogate_alg is not None:
        cli_params['surrogate_alg_name'] = surrogate_alg

    config = {
        **config,
        **cli_params
    }

    # TODO: log all params

    df = pd.read_csv(in_)

    with prepare_working_dir(working_dir) as work_dir:
        autotm = AutoTM(
            **config,
            working_dir_path=work_dir
        )
        mixtures = autotm.fit_predict(df)

        logger.info(f"Calculated train mixtures: {mixtures.shape}\n\n{mixtures.head(10).to_string()}")

        # saving artifacts
        autotm.save(model)
        mixtures.to_csv(out)

    click.echo('Initialized the database')


@cli.command()
def predict():
    click.echo('Dropped the database')


if __name__ == "__main__":
    cli()
