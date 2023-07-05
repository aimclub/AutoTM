FROM catthehacker/ubuntu:act-latest

RUN apt update && apt install -y python3-pip

COPY . /src

WORKDIR /src

RUN python -m pip install --upgrade pip
RUN python -m pip install flake8 pytest poetry
RUN poetry export -f requirements.txt --without-hashes --with dev > requirements.txt
RUN pip install --ignore-installed -r requirements.txt