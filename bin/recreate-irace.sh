#!/usr/bin/env bash

set -ex

./bin/build-docker.sh
./bin/push-docker-to-repo.sh
./bin/stop-irace.sh
./bin/start-irace.sh
