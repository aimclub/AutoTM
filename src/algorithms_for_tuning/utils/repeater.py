import asyncio
import datetime
import glob
import logging
import os
import random
import shutil
import sys
import uuid
from asyncio import Future
from asyncio.tasks import FIRST_COMPLETED, ALL_COMPLETED
from logging import config
from typing import List, Tuple, Iterable, Set
from typing import Optional

import click
import yaml

DATETIME_FORMAT = "%Y-%m-%dT%H-%M-%S"
FULL_RECORD_SYMBOL = "END"


def get_logging_config(logfile: str):
    return {
        'version': 1,
        'disable_existing_loggers': True,
        'formatters': {
            'standard': {
                'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            },
        },
        'handlers': {
            'default': {
                'level': 'DEBUG',
                'formatter': 'standard',
                'class': 'logging.StreamHandler',
                'stream': 'ext://sys.stderr',
            },
            'logfile': {
                'level': 'DEBUG',
                'formatter': 'standard',
                'class': 'logging.FileHandler',
                'filename': f'{logfile}',
            }
        },
        'loggers': {
            'root': {
                'handlers': ['default'],
                'level': 'DEBUG',
                'propagate': False
            },
            'REPEATER': {
                'handlers': ['default', 'logfile'],
                'level': 'DEBUG',
                'propagate': False
            }
        }
    }


logger = logging.getLogger("REPEATER")


class Repeater:
    @staticmethod
    def _get_checkpoint_record(rep_num, cmd: str, args: List[str]) -> str:
        return f"{rep_num}\t{cmd}\t{' '.join(args)}\t{FULL_RECORD_SYMBOL}"

    @staticmethod
    def _check_for_exceptions(done: Set[Future]) -> None:
        for d in done:
            if d.exception():
                logger.error("Found error in coroutines of processes.", exc_info=d.exception())
                # raise d.exception()

    def __init__(self, cfg: dict, checkpoint_path: Optional[str], run_tag: Optional[str] = None):
        self.cfg = cfg
        self.checkpoint_path = checkpoint_path
        self.run_tag = run_tag if run_tag is not None else str(uuid.uuid4())
        logger.info(f"Running with RUN TAG: {self.run_tag}")

    def _load_and_prepare_checkpoint(self, previous_checkpoint_path: Optional[str]) -> Set[str]:
        logger.info(f"Trying to load a checkpoint if possible. Previous checkpoint path: {previous_checkpoint_path}")
        if self.checkpoint_path and previous_checkpoint_path:
            if not os.path.exists(previous_checkpoint_path):
                logger.warning(f"Previous checkpoint file was not found on path {previous_checkpoint_path}. "
                               f"Continue without reading and copying its content")
            else:
                logger.info(f"Copying previous checkpoint from {previous_checkpoint_path} to {self.checkpoint_path}. "
                            f"It will be a start point for a new run.")
                shutil.copy(previous_checkpoint_path, self.checkpoint_path, follow_symlinks=True)

        if os.path.exists(self.checkpoint_path):
            logger.info(f"Loading checkpoint data from {self.checkpoint_path}")
            with open(self.checkpoint_path, "r") as f:
                checkpoint = set(line.strip() for line in f.readlines() if line.strip().endswith(FULL_RECORD_SYMBOL))
        else:
            checkpoint = set()

        return checkpoint

    def _prepare_configurations(self, previous_checkpoint_path: Optional[str]) \
            -> Iterable[Tuple[int, str, str, List[str]]]:
        logger.info("Preparing the list of configurations for the run")
        datasets: List[str] = self.cfg["datasets"]
        alg_configs: List[dict] = self.cfg["configurations"]

        checkpoint = self._load_and_prepare_checkpoint(previous_checkpoint_path)

        for dataset in datasets:
            for alg_cfg in alg_configs:
                cmd, args, workdir, repetitions = alg_cfg["cmd"], alg_cfg["args"], alg_cfg["workdir"], alg_cfg["repetitions"]
                args = ["--dataset", dataset, "--tag", self.run_tag] + args.split(" ")
                for i in range(repetitions):
                    record = self._get_checkpoint_record(i, cmd, args)
                    if record in checkpoint:
                        logger.info(f"Found configuration '{record}' in checkpoint. Skipping.")
                    else:
                        yield i, cmd, workdir, args

    def _save_to_checkpoint(self, rep_num, cmd: str, args: List[str]) -> None:
        if self.checkpoint_path:
            with open(self.checkpoint_path, "a") as f:
                f.write(self._get_checkpoint_record(rep_num, cmd, args) + "\n")

    async def _execute_run(self, rep_num: int, cmd: str, workdir: str, args: List[str]) -> None:
        proc = await asyncio.create_subprocess_exec(cmd, *args,
                                                    stdout=sys.stdout, stderr=sys.stdout, cwd=workdir)
        ret_code = await proc.wait()

        if ret_code != 0:
            msg = f"Return code {ret_code} != 0 for run (repetition {rep_num}) with cmd '{cmd}' and args '{args}'"
            logger.error(msg)
            raise Exception(msg)
        else:
            logger.info(f"Successful run with cmd '{cmd}' and args '{args}'")
            self._save_to_checkpoint(rep_num, cmd, args)

    async def run_repetitions(self, previous_checkpoint_path, max_parallel_processes: Optional[int] = None):
        logger.info("Starting the run")
        configurations = list(self._prepare_configurations(previous_checkpoint_path))
        configurations = random.sample(configurations, len(configurations))
        # processes = (self._execute_run(rep_num, cmd, workdir, args) for rep_num, cmd, workdir, args in configurations)
        processes = [self._execute_run(rep_num, cmd, workdir, args) for rep_num, cmd, workdir, args in configurations]
        logger.info(f"Initial number of configurations to calculate: {len(configurations)}")

        if len(configurations) == 0:
            logger.warning(f"No configurations to calculate. Interrupting execution.")
            return

        if max_parallel_processes:
            logger.info(f"Max count of parallel processes are restricted to {max_parallel_processes}")

            total_done_count = 0
            run_slots: List[Future]  = [asyncio.create_task(p) for p in processes[:max_parallel_processes]]
            processes = processes[max_parallel_processes:] if len(processes) > max_parallel_processes else []
            while len(run_slots) > 0:
                done, pending = await asyncio.wait(run_slots, return_when=FIRST_COMPLETED)
                self._check_for_exceptions(done)
                total_done_count += len(done)
                free_slots = max_parallel_processes - len(pending)
                run_slots = list(pending) + processes[:free_slots]
                processes = processes[free_slots:] if len(processes) > free_slots else []
                logger.info(f"{total_done_count} configurations have been calculated. "
                            f"{len(configurations) - total_done_count} are left.")
        else:
            logger.info(f"No restrictions on number of parallel processes. "
                        f"Starting all {len(configurations)} configurations.")
            await asyncio.wait(processes)


def extract_datetime(filename: str) -> datetime.datetime:
    name, ext = os.path.splitext(os.path.basename(filename))
    dt_str = name.split('_')[-1]
    return datetime.datetime.strptime(dt_str, DATETIME_FORMAT)


def find_checkpoints(checkpoint_dir: str, checkpoint_prefix: str) -> Tuple[str, str]:
    logger.info(f"Looking for the previous checkpoint file in {checkpoint_dir}")

    files = glob.glob(f"{checkpoint_dir}/{checkpoint_prefix}_*.txt")
    files = sorted(files, key=lambda x: extract_datetime(x), reverse=True)
    previous_checkpoint_file = files[0] if len(files) > 0 else None
    cur_dt = datetime.datetime.now().strftime(DATETIME_FORMAT)
    checkpoint_file = f"{checkpoint_dir}/{checkpoint_prefix}_{cur_dt}.txt"

    return previous_checkpoint_file, checkpoint_file


@click.command(context_settings=dict(allow_extra_args=True))
@click.option('--config', 'yaml_config', required=True, help='a path to the config file', type=str)
@click.option('--checkpoint-dir', required=False, help='a path to the directory where checkpoints is stored', type=str)
@click.option('--checkpoint-prefix',
              required=False, default="repeater-checkpoint", help='a prefix to be used in checkpoint files', type=str)
@click.option('--parallel',
              required=False, help='a max number of parallel processes running at the same moment', type=int)
@click.option('--log-file', default="/var/log/repeator.log",
              help='a log file to write logs of the algorithm execution to')
@click.option('--tag', required=False, type=str,
              help='a custom tag string to denote the current set of experiments')
def main(yaml_config: str,
         checkpoint_dir: Optional[str],
         checkpoint_prefix: Optional[str],
         parallel: Optional[int],
         log_file: str,
         tag: Optional[str]):
    logging.config.dictConfig(get_logging_config(log_file))

    logger.info(f"Starting repeater with arguments: {sys.argv}")
    with open(yaml_config, "r") as f:
        cfg = yaml.load(f)

    previous_checkpoint_file, checkpoint_file = \
        find_checkpoints(checkpoint_dir, checkpoint_prefix) if checkpoint_dir else (None, None)

    r = Repeater(cfg, checkpoint_file, run_tag=tag)
    asyncio.run(r.run_repetitions(previous_checkpoint_file, max_parallel_processes=parallel))
    logger.info("Repeater has finished.")


if __name__ == "__main__":
    main()
