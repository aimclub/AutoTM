#!/usr/bin/env bash

export PYTHONPATH="./:./algorithms_for_tuning:${PYTHONPATH}"
exec python3 ./algorithms_for_tuning/utils/repeater.py "${@}"
