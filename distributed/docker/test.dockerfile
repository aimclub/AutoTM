FROM ubuntu:20.04
RUN apt-get update && apt-get install -y bash python3 python3-pip
RUN pip3 install  'celery == 4.4.7' 'bigartm == 0.9.2' 'tqdm == 4.50.2' 'numpy == 1.19.2' 'dataclasses-json == 0.5.2'