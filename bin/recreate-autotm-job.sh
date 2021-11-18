#!/usr/bin/env bash

set -ex

./bin/build-docker.sh
./bin/push-docker-to-repo.sh
./bin/stop-autotm-job.sh
./bin/start-autotm-job.sh
