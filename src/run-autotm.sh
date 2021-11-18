#!/usr/bin/env bash

echo "Arguments: "
echo "${@}"

python3 ./algorithms_for_tuning/genetic_algorithm/genetic_algorithm.py "${@}"
