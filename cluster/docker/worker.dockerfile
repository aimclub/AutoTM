FROM ubuntu:20.04

RUN apt-get update && apt-get install -y bash python3 python3-pip

#RUN pip3 install  'celery == 4.4.7' 'bigartm == 0.9.2' 'tqdm == 4.50.2' 'numpy == 1.19.2' 'dataclasses-json == 0.5.2'

COPY requirements.txt /tmp/

RUN pip3 install -r  /tmp/requirements.txt

COPY dist/autotm-0.1.0-py3-none-any.whl /tmp

RUN pip3 install --no-deps /tmp/autotm-0.1.0-py3-none-any.whl

ENTRYPOINT fitness-worker \
    --concurrency 1 \
    --queues fitness_tasks \
    --loglevel INFO
