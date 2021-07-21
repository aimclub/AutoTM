#!/usr/bin/env bash

export PYTHONPATH="../../:../../algorithms_for_tuning:${PYTHONPATH}"
exec python3 ../../algorithms_for_tuning/ga_with_surrogate/ga_with_surrogate_through_iteration.py "${@}"