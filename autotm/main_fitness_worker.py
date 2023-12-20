import os

import yaml


def main():
    os.environ['AUTOTM_COMPONENT'] = 'worker'
    os.environ['AUTOTM_EXEC_MODE'] = 'cluster'

    from autotm.fitness import make_celery_app
    from autotm.fitness.tm import TopicModelFactory

    if "DATASETS_CONFIG" in os.environ:
        with open(os.environ["DATASETS_CONFIG"], "r") as f:
            config = yaml.safe_load(f)
            dataset_settings = config["datasets"]
    else:
        dataset_settings = None

    TopicModelFactory.init_factory_settings(
        num_processors=os.getenv("NUM_PROCESSORS", None),
        dataset_settings=dataset_settings
    )

    celery_app = make_celery_app()
    celery_app.worker_main()


if __name__ == '__main__':
    main()
