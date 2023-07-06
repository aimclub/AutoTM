FROM python:3.8

RUN pip install PyMySQL==0.9.3 psycopg2-binary==2.8.5 protobuf==3.20.1 mlflow[extras]==1.18.0

ENTRYPOINT ["mlflow", "server"]
