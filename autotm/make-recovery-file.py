#!/usr/bin/python3
import datetime
import logging
import os
import shutil
import sys
from glob import glob

logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger()


# example: irace-log-2021-07-19T21:48:00-2b5faba3-99a8-47ba-a5cd-2c99e6877160.Rdata
def fpath_to_datetime(fpath: str) -> datetime:
    fname = os.path.basename(fpath)
    dt_str = fname[len("irace-log-"): len("irace-log-") + len("YYYY-mm-ddTHH:MM:ss")]
    dt = datetime.datetime.strptime(dt_str, "%Y-%m-%dT%H:%M:%S")
    return dt


if __name__ == "__main__":
    save_dir = sys.argv[1]
    logger.info(f"Looking for data files in {save_dir}")
    data_files = [
        (fpath, fpath_to_datetime(fpath)) for fpath in glob(f"{save_dir}/*.Rdata")
    ]

    if len(data_files) == 0:
        logger.info(
            f"No data files have been found in {save_dir}. It is a normal situation. Interrupting execution."
        )
        sys.exit(0)

    last_data_filepath, _ = max(data_files, key=lambda x: x[1])
    recovery_filepath = os.path.join(save_dir, "recovery_rdata.checkpoint")

    logger.info(
        f"Copying last generated data file ({last_data_filepath}) as recovery file ({recovery_filepath})"
    )
    shutil.copy(last_data_filepath, recovery_filepath)
    logger.info("Copying done")
