FROM python:3.9

RUN apt-get update && apt-get install -y bash python3 python3-pip

COPY requirements.txt /tmp/

RUN pip3 install -r  /tmp/requirements.txt

COPY dist/autotm-0.1.0-py3-none-any.whl /tmp

RUN pip3 install --no-deps /tmp/autotm-0.1.0-py3-none-any.whl

RUN pip3 install jupyter

RUN pip3 install datasets

RUN python -m spacy download en_core_web_sm

COPY data /root

COPY examples/autotm_demo_updated.ipynb /root

ENTRYPOINT ["jupyter", "notebook", "--notebook-dir=/root", "--ip=0.0.0.0", "--port=8888", "--allow-root", "--no-browser", "--NotebookApp.token=''"]
