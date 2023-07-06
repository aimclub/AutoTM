FROM fitness-base:latest
ENTRYPOINT ["python3", "-u", "/kube-distributed-fitness/kube_fitness/test_app.py"]