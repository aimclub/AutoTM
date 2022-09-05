#!/usr/bin/env bash

set -e

rm -rf /tmp/repeater/
mkdir -p /tmp/repeater/
mkdir -p /tmp/repeater/logs

exec python3 ../../src/algorithms_for_tuning/utils/repeater.py \
  --config mock-repeater-config.yaml \
  --checkpoint-dir /tmp/repeater/ \
  --runs-log-dir /tmp/repeater/logs \
  --log-file /tmp/repeater/repeater.log \
  --parallel 20
