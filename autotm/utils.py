import os
import io
import logging
from datetime import datetime
from typing import Optional
import numpy as np
import pandas as pd
from multiprocessing import Pool
import multiprocessing as mp
from functools import partial

from typing import Dict, Any

from collections import Counter

MetricsScores = Dict[str, Any]
TimeMeasurements = Dict[str, float]
AVG_COHERENCE_SCORE = "avg_coherence_score"

logger = logging.getLogger(__name__)


class TqdmToLogger(io.StringIO):
    """
    Output stream for TQDM which will output to logger module instead of
    the StdOut.
    """

    def __init__(self, base_logger, level=None):
        super(TqdmToLogger, self).__init__()
        self.logger = base_logger
        self.level = level or logging.INFO
        self.buf = ""

    def write(self, buf):
        self.buf = buf.strip("\r\n\t ")

    def flush(self):
        self.logger.log(self.level, self.buf)


class log_exec_timer:
    def __init__(self, name: Optional[str] = None):
        self.name = name
        self._start = None
        self._duration = None

    def __enter__(self):
        self._start = datetime.now()
        return self

    def __exit__(self, typ, value, traceback):
        self._duration = (datetime.now() - self._start).total_seconds()
        msg = (
            f"Exec time of {self.name}: {self._duration}"
            if self.name
            else f"Exec time: {self._duration}"
        )
        logger.info(msg)

    @property
    def duration(self):
        return self._duration


def merge_dicts(dicts):
    full_dict = {}
    for d in dicts:
        full_dict = dict(Counter(full_dict) + Counter(d))
    return full_dict


def parallelize_dataframe(df: pd.DataFrame, func, n_cores, return_type="df", **kwargs):
    """

    :param df: Dataframe to process.
    :param func: Function to be applied in parallel mode on data chunks
    :param n_cores: Amount of cores to parallelize on. In case of -1 takes all the available cores.
    :param return_type: datatype returned by func: 'df' or 'dict'
    :param kwargs: Additional parameters of the function, which is applied in parallel mode.
    :return: pd.DataFrame
    """
    if n_cores == -1:
        n_cores = mp.cpu_count() - 1
    df_split = np.array_split(df, n_cores)

    pool = Pool(n_cores)
    func_with_args = partial(func, **kwargs)
    map_res = pool.map(func_with_args, df_split)
    if return_type == "df":
        if isinstance(map_res[0], pd.DataFrame):
            res = pd.concat(map_res)
        elif isinstance(map_res[0], tuple):
            zipped_elems = list(zip(*map_res))
            res = (pd.concat(zipped_elems[0]), pd.concat(zipped_elems[1]))
    elif return_type == "dict":
        if isinstance(map_res[0], dict):
            res = merge_dicts(map_res)
        elif isinstance(map_res[0], tuple):
            zipped_elems = list(zip(*map_res))
            res = (merge_dicts(zipped_elems[0]), merge_dicts(zipped_elems[1]))
    pool.close()
    pool.join()
    return res


def make_log_config_dict(
    filename: str = "/var/log/tm-alg.txt", uid: Optional[str] = None
) -> Dict[str, Any]:
    if filename is not None:
        if uid:
            dirname = os.path.dirname(filename)
            file, ext = os.path.splitext(os.path.basename(filename))
            log_filename = os.path.join(dirname, f"{file}-{uid}.{ext}")
        else:
            log_filename = filename

        logfile_handler = {
            "logfile": {
                "level": "DEBUG",
                "formatter": "standard",
                "class": "logging.FileHandler",
                "filename": log_filename,
            }
        }
        handlers = ["default", "logfile"]
    else:
        logfile_handler = dict()
        handlers = ["default"]

    return {
        "version": 1,
        "disable_existing_loggers": True,
        "formatters": {
            "standard": {
                "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            },
        },
        "handlers": {
            "default": {
                "level": "DEBUG",
                "formatter": "standard",
                "class": "logging.StreamHandler",
                "stream": "ext://sys.stderr",
            },
            **logfile_handler
        },
        "loggers": {
            "root": {
                "handlers": handlers,
                "level": "DEBUG",
                "propagate": False,
            },
            "GA": {
                "handlers": handlers,
                "level": "DEBUG",
                "propagate": False,
            },
            "GA_algo": {
                "handlers": handlers,
                "level": "DEBUG",
                "propagate": False,
            },
            "GA_surrogate": {
                "handlers": handlers,
                "level": "DEBUG",
                "propagate": False,
            },
        },
    }
