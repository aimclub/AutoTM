FROM fitness-base:latest
ENTRYPOINT python3 /kube-distributed-fitness/kube_fitness/kube_fitness/main.py \
    --concurrency 1 \
    --queues fitness_tasks \
    --loglevel INFO