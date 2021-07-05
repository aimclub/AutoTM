#!/usr/bin/env bash

set -ex

docker build -t irace-runner:latest -f docker/irace-runner.dockerfile .
