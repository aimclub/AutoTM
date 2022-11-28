import io
import logging
from datetime import datetime
from typing import Optional
import numpy as np
import pandas as pd
from multiprocessing import Pool
import multiprocessing as mp
from functools import partial

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


def parallelize_dataframe(df: pd.DataFrame, func, n_cores, **kwargs):
    '''

    :param df: Dataframe to process.
    :param func: Function to be applied in parallel mode on data chunks
    :param n_cores: Amount of cores to parallelize on. In case of -1 takes all the available cores.
    :param kwargs: Additional parameters of the function, which is applied in parallel mode.
    :return: pd.DataFrame
    '''
    if n_cores == -1:
        n_cores = mp.cpu_count()
    df_split = np.array_split(df, n_cores)
    pool = Pool(n_cores)
    func_with_args = partial(func, **kwargs)
    df = pd.concat(pool.map(func_with_args, df_split))
    pool.close()
    pool.join()
    return df
