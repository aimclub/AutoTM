#!/usr/bin/env bash

set -ex

echo "Input arguments: [" "${@}" "]"

./bin/stop-repeater.sh "${@}"
./bin/build-docker.sh
./bin/push-docker-to-repo.sh
./bin/start-repeater.sh "${@}"
