#!/usr/bin/env bash

set -e

export KUBE_NAMESPACE=mkhodorchenko

./bin/build-docker.sh
./bin/push-docker-to-repo.sh
./bin/stop-repeater.sh
./bin/start-repeater.sh
