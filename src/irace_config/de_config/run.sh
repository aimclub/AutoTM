#!/usr/bin/env bash

export PYTHONPATH="../../:../../algorithms_for_tuning:${PYTHONPATH}"
exec python3 ../../algorithms_for_tuning/de_algorithm/de.py "${@}"