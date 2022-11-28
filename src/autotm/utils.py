import io
import logging
from datetime import datetime
from typing import Optional
import numpy as np
import pandas as pd
from multiprocessing import Pool
from itertools import repeat

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


def parallelize_dataframe(df, func, n_cores=4):
    df_split = np.array_split(df, n_cores)
    pool = Pool(n_cores)
    df = pd.concat(pool.map(func, df_split))
    pool.close()
    pool.join()
    return df
