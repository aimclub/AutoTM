FROM ubuntu:18.04

RUN DEBIAN_FRONTEND=noninteractive apt-get update -y && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y git curl make cmake build-essential libboost-all-dev

RUN curl -LJO https://github.com/bigartm/bigartm/archive/refs/tags/v0.9.2.tar.gz

RUN tar -xvf bigartm-0.9.2.tar.gz

RUN mkdir bigartm-0.9.2/build

RUN DEBIAN_FRONTEND=noninteractive apt-get update && DEBIAN_FRONTEND=noninteractive apt-get --yes install bash python3.8

RUN curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py

RUN python3.8 get-pip.py

RUN pip3 install protobuf tqdm wheel

RUN cd bigartm-0.9.2/build && cmake .. -DBUILD_INTERNAL_PYTHON_API=OFF

RUN cd bigartm-0.9.2/build && make install

COPY requirements.txt /tmp/

RUN pip3 install -r  /tmp/requirements.txt

COPY . /kube-distributed-fitness

RUN pip3 install /kube-distributed-fitness
