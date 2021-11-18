FROM irace-base:latest

WORKDIR /app

ENTRYPOINT ["/app/run-autotm.sh"]