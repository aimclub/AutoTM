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
        self.buf = ''

    def write(self, buf):
        self.buf = buf.strip('\r\n\t ')

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
        msg = f"Exec time of {self.name}: {self._duration}" if self.name else f"Exec time: {self._duration}"
        logger.info(msg)

    @property
    def duration(self):
        return self._duration


def merge_dicts(dicts):
    full_dict = {}
    for d in dicts:
        full_dict = dict(Counter(full_dict) + Counter(d))
    return full_dict


def parallelize_dataframe(df: pd.DataFrame, func, n_cores, return_type='df', **kwargs):
    '''

    :param df: Dataframe to process.
    :param func: Function to be applied in parallel mode on data chunks
    :param n_cores: Amount of cores to parallelize on. In case of -1 takes all the available cores.
    :param return_type:
    :param kwargs: Additional parameters of the function, which is applied in parallel mode.
    :return: pd.DataFrame
    '''
    if n_cores == -1:
        n_cores = mp.cpu_count()
    df_split = np.array_split(df, n_cores)
    pool = Pool(n_cores)
    func_with_args = partial(func, **kwargs)
    result = pool.map(func_with_args, df_split)
    if return_type == 'df':
        df = pd.concat(pool.map(func_with_args, df_split))
    else:
        df = pd.concat(pool.map(func_with_args, df_split))
    pool.close()
    pool.join()
    return df
