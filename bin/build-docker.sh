#!/usr/bin/env bash

set -ex

docker build -t irace-base:latest -f docker/irace-base.dockerfile .
docker build -t irace-runner:latest -f docker/irace-runner.dockerfile .
docker build -t exp-repeater:latest -f docker/exp-repeater.dockerfile .
docker build -t autotm-job:latest -f docker/autotm-job.dockerfile .
