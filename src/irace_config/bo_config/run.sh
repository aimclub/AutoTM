#!/usr/bin/env bash

export PYTHONPATH="../../:../../algorithms_for_tuning:${PYTHONPATH}"
exec python3 ../../algorithms_for_tuning/bo_algorithm/bayes_opt.py "${@}"