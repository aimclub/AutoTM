#!/usr/bin/env bash

set -ex

docker tag irace-runner:latest node2.bdcl:5000/irace-runner:latest
docker push node2.bdcl:5000/irace-runner:latest
