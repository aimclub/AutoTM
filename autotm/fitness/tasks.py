import functools

from . import AUTOTM_EXEC_MODE, SUPPORTED_EXEC_MODES

if AUTOTM_EXEC_MODE == 'local':
    from .local_tasks import estimate_fitness, log_best_solution
elif AUTOTM_EXEC_MODE == 'cluster':
    from .cluster_tasks import make_celery_app
    from .cluster_tasks import parallel_fitness, log_best_solution

    app = make_celery_app()
    estimate_fitness = functools.partial(parallel_fitness, app=app)
    log_best_solution = functools.partial(log_best_solution, app=app)
else:
    raise ValueError(f"Unknown exec mode: {AUTOTM_EXEC_MODE}. Only the following are supported: {SUPPORTED_EXEC_MODES}")
