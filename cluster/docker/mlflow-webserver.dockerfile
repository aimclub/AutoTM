FROM python:3.8

RUN pip install PyMySQL==0.9.3 psycopg2-binary==2.9.9 protobuf==3.20.0 mlflow[extras]==2.9.2

ENTRYPOINT ["mlflow", "server"]
