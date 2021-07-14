import asyncio
import copy
import json
import logging
import random
import sys
from typing import List, Tuple, Iterable, Optional

import click
import yaml

logger = logging.getLogger("REPEATER")


class Repeater:
    @staticmethod
    def load_checkpoint(cfg: dict, checkpoint_path: Optional[str], new_checkpoint_path: Optional[str]):
        if checkpoint_path:
            with open(checkpoint_path, "r") as f:
                checkpoint_log = f.readlines()
        else:
            checkpoint_log = None

        return Repeater(cfg, checkpoint_log, new_checkpoint_path)

    @staticmethod
    def _get_checkpoint_record(rep_num, cmd: str, args: List[str]) -> str:
        return f"{rep_num}]\t{cmd}\t{' '.join(args)}"

    def __init__(self, cfg: dict, checkpoint_log: Optional[List[str]], checkpoint_path: Optional[str]):
        self.cfg = cfg
        self.checkpoint = set(checkpoint_log) if checkpoint_log else set()
        self.checkpoint_path = checkpoint_path

    def _make_parameters(self) -> Iterable[Tuple[int, str, List[str]]]:
        datasets: List[str] = self.cfg["datasets"]
        alg_configs: List[dict] = self.cfg["configurations"]

        for dataset in datasets:
            for alg_cfg in alg_configs:
                cmd, args, repetitions = alg_cfg["cmd"], alg_cfg["args"], alg_cfg["repetitions"]
                args = ["--dataset", dataset] + args.split(" ")
                for i in range(repetitions):
                    record = self._get_checkpoint_record(i, cmd, args)
                    if record in self.checkpoint:
                        logger.info(f"Found configuration '{record}' in checkpoint. Skipping")
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

    async def run_repetitions(self):
        parameters = list(self._make_parameters())
        parameters = random.sample(parameters, len(parameters))
        processes = (self._execute_run(rep_num, cmd, args) for rep_num, cmd, args in parameters)
        await asyncio.wait(processes)


@click.command(context_settings=dict(allow_extra_args=True))
@click.option('--config', required=True, help='a path to the config file', type=str)
@click.option('--checkpoint', required=False, help='a path to the config file', type=str)
def main(config: str, checkpoint: Optional[str]):
    with open(config, "r") as f:
        cfg = yaml.load(f)

    # TODO: add logic to work with checkpoints
    r = Repeater(cfg, checkpoint)

    asyncio.run(r.run_repetitions())








if __name__ == "__main__":
    main()