import os

import yaml

from kube_fitness.tasks import make_celery_app
from kube_fitness.tm import TopicModelFactory

if __name__ == '__main__':
    if "DATASETS_CONFIG" in os.environ:
        with open(os.environ["DATASETS_CONFIG"], "r") as f:
            config = yaml.load(f)
            dataset_settings = config["datasets"]
    else:
        dataset_settings = None

    TopicModelFactory.init_factory_settings(
        num_processors=os.getenv("NUM_PROCESSORS", None),
        dataset_settings=dataset_settings
    )

    celery_app = make_celery_app()
    celery_app.worker_main()
