#!/usr/bin/env bash

docker run -it --rm \
 -v /home/nikolay/wspace/20newsgroups/:/dataset \
 -e "CELERY_BROKER_URL=amqp://guest:guest@rabbitmq-service:5672" \
 -e "CELERY_RESULT_BACKEND=rpc://" \
 -e "DICTIONARY_PATH=/dataset/dictionary.txt" \
 -e "BATCHES_DIR=/dataset/batches" \
 -e "MUTUAL_INFO_DICT_PATH=/dataset/mutual_info_dict.pkl" \
 -e "EXPERIMENTS_PATH=/tmp/tm_experiments" \
 -e "TOPIC_COUNT=10" \
 -e "NUM_PROCESSORS=1" \
 fitness-worker:latest h