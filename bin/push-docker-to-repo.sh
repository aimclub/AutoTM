#!/usr/bin/env bash

set -ex

docker tag irace-runner:latest node2.bdcl:5000/irace-runner:latest
docker push node2.bdcl:5000/irace-runner:latest

docker tag exp-repeater:latest node2.bdcl:5000/exp-repeater:latest
docker push node2.bdcl:5000/exp-repeater:latest

docker tag autotm-job:latest node2.bdcl:5000/autotm-job:latest
docker push node2.bdcl:5000/autotm-job:latest
