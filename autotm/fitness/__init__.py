import functools
import os

# local or cluster
SUPPORTED_EXEC_MODES = ['local', 'cluster']
AUTOTM_EXEC_MODE = os.environ.get("AUTOTM_EXEC_MODE", "local")


# head or worker
SUPPORTED_COMPONENTS = ['head', 'worker']
AUTOTM_COMPONENT = os.environ.get("AUTOTM_COMPONENT", "head")


if AUTOTM_EXEC_MODE == 'local':
    from .tasks import estimate_fitness, log_best_solution
elif AUTOTM_EXEC_MODE == 'cluster':
    from .cluster_tasks import make_celery_app
    from .cluster_tasks import parallel_fitness, log_best_solution

    app = make_celery_app()
    estimate_fitness = functools.partial(parallel_fitness, app=app)
    log_best_solution = functools.partial(log_best_solution, app=app)
else:
    raise ValueError(f"Unknown exec mode: {AUTOTM_EXEC_MODE}. Only the following are supported: {SUPPORTED_EXEC_MODES}")
