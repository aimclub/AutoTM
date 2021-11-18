FROM irace-base:latest

WORKDIR /app

ENV PYTHONPATH=/app:$PYTHONPATH

ENTRYPOINT ["/app/run-autotm.sh"]