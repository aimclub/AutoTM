import asyncio
import datetime
import glob
import logging
import os
import random
import shutil
import sys
from typing import List, Tuple, Iterable, Optional, Set

import click
import yaml

DATETIME_FORMAT = "%Y-%m-%dT%H-%M-%S"
FULL_RECORD_SYMBOL = "END"

logger = logging.getLogger("REPEATER")


class Repeater:
    @staticmethod
    def _get_checkpoint_record(rep_num, cmd: str, args: List[str]) -> str:
        return f"{rep_num}]\t{cmd}\t{' '.join(args)}\t{FULL_RECORD_SYMBOL}"

    def __init__(self, cfg: dict, checkpoint_path: Optional[str]):
        self.cfg = cfg
        self.checkpoint_path = checkpoint_path

    def _load_and_prepare_checkpoint(self, previous_checkpoint_path: Optional[str]) -> Set[str]:
        if self.checkpoint_path and previous_checkpoint_path:
            if os.path.exists(previous_checkpoint_path):
                logger.warning(f"Previous checkpoint file was not found on path {previous_checkpoint_path}. "
                               f"Continue without reading and copying its content")
            else:
                shutil.copy(previous_checkpoint_path, self.checkpoint_path, follow_symlinks=True)

        if os.path.exists(self.checkpoint_path):
            with open(self.checkpoint_path, "r") as f:
                checkpoint = set(line.strip() for line in f.readlines() if line.endswith(FULL_RECORD_SYMBOL))
        else:
            checkpoint = set()

        return checkpoint

    def _make_parameters(self, previous_checkpoint_path: Optional[str]) -> Iterable[Tuple[int, str, List[str]]]:
        datasets: List[str] = self.cfg["datasets"]
        alg_configs: List[dict] = self.cfg["configurations"]

        checkpoint = self._load_and_prepare_checkpoint(previous_checkpoint_path)

        for dataset in datasets:
            for alg_cfg in alg_configs:
                cmd, args, repetitions = alg_cfg["cmd"], alg_cfg["args"], alg_cfg["repetitions"]
                args = ["--dataset", dataset] + args.split(" ")
                for i in range(repetitions):
                    record = self._get_checkpoint_record(i, cmd, args)
                    if record in checkpoint:
                        logger.info(f"Found configuration '{record}' in checkpoint. Skipping.")
                    else:
                        yield i, cmd, args

    def _save_to_checkpoint(self, rep_num, cmd: str, args: List[str]) -> None:
        if self.checkpoint_path:
            with open(self.checkpoint_path, "a") as f:
                f.write(self._get_checkpoint_record(rep_num, cmd, args))

    async def _execute_run(self, rep_num: int, cmd: str, args: List[str]) -> None:
        proc = await asyncio.create_subprocess_exec(program=cmd, args=args, stdout=sys.stdout, stderr=sys.stdout)
        ret_code = await proc.wait()
        if ret_code != 0:
            msg = f"Return code {ret_code} != 0 for run (repetition {rep_num}) with cmd '{cmd}' and args '{args}'"
            logger.error(msg)
            raise Exception(msg)
        else:
            logger.info(f"Successful run with cmd '{cmd}' and args '{args}'")
            self._save_to_checkpoint(rep_num, cmd, args)

    async def run_repetitions(self, previous_checkpoint_path):
        parameters = list(self._make_parameters(previous_checkpoint_path))
        parameters = random.sample(parameters, len(parameters))
        processes = (self._execute_run(rep_num, cmd, args) for rep_num, cmd, args in parameters)
        await asyncio.wait(processes)


def extract_datetime(filename: str) -> datetime.datetime:
    name, ext = os.path.splitext(os.path.basename(filename))
    dt_str = name.split('_')[-1]
    return datetime.datetime.strptime(dt_str, DATETIME_FORMAT)


@click.command(context_settings=dict(allow_extra_args=True))
@click.option('--config', required=True, help='a path to the config file', type=str)
@click.option('--checkpoint-dir', required=False, help='a path to the config file', type=str)
@click.option('--checkpoint-prefix', required=False, default="checkpoint", help='a path to the config file', type=str)
def main(config: str, checkpoint_dir: Optional[str], checkpoint_prefix: Optional[str]):
    with open(config, "r") as f:
        cfg = yaml.load(f)

    if checkpoint_dir:
        files = glob.glob(f"{checkpoint_dir}/{checkpoint_prefix}_*.txt")
        files = sorted(files, key=lambda x: extract_datetime(x), reverse=True)
        previous_checkpoint_file = files[1] if len(files) > 0 else None
        cur_dt = datetime.datetime.now().strftime(DATETIME_FORMAT)
        checkpoint_file = f"{checkpoint_dir}/{checkpoint_prefix}_{cur_dt}.txt"
    else:
        previous_checkpoint_file, checkpoint_file = None, None

    r = Repeater(cfg, checkpoint_file)
    asyncio.run(r.run_repetitions(previous_checkpoint_file))


if __name__ == "__main__":
    main()
