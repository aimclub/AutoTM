import logging
from pprint import pprint
from typing import Optional

import click

from kube_fitness.metrics import AVG_COHERENCE_SCORE

logging.basicConfig(level=logging.DEBUG)

from kube_fitness.tasks import make_celery_app, parallel_fitness, log_best_solution
from kube_fitness.schemas import IndividualDTO

logger = logging.getLogger("TEST_APP")

celery_app = make_celery_app(client_to_broker_max_retries=25)

population = [
    IndividualDTO(id="1", dataset="first_dataset", topic_count=5, exp_id=0, alg_id="test_alg", iteration_id=100,
        tag="just_a_simple_tag", params=[
            66717.86348968784, 2.0, 98.42286785825902,
            80.18543570807961, 2.0, -19.948347560420373,
            -52.28141634493725, 2.0, -92.85597392137976,
            -60.49287378084627, 4.0, 3.0, 0.06840138630839943,
            0.556001061599461, 0.9894122432621849, 11679.364068753106]),

    IndividualDTO(id="2", dataset="second_dataset", topic_count=5, params=[
        42260.028918433134, 1.0, 7.535922806674758,
        92.32509683092258, 3.0, -71.92218883101997,
        -56.016307418098386, 1.0, -3.237446735558109,
        -20.448661743825, 4.0, 7.0, 0.08031597367372223,
        0.42253357962287563, 0.02249898631530911, 61969.93405537756]),

    IndividualDTO(id="3", dataset="first_dataset", topic_count=5, params=[
        53787.42158965788, 1.0, 96.00600751876978,
        82.83724552058459, 6.0, -51.625141715571715,
        -35.45911388077616, 0.0, -79.6738397452075,
        -42.96576312232228, 3.0, 6.0, 0.1842678599143196,
        0.08563438015417912, 0.104280507307428, 91187.26051038165]),

    IndividualDTO(id="4", dataset="second_dataset", topic_count=5, params=[
        24924.136392296103, 0.0, 98.63988602807903,
        49.03544407815009, 5.0, -8.734591095928806,
        -6.99720964952175, 4.0, -32.880078901677265,
        -24.61400511189416, 3.0, 0.0, 0.9084621726817743,
        0.6392049950522389, 0.3133878344721094, 39413.00378611856])
]


@click.group()
def cli():
    pass


@cli.command()
def test():
    logger.info("Starting parallel computations...")

    population_with_fitness = parallel_fitness(population)

    assert len(population_with_fitness) == 4, \
        f"Wrong population size (should be == 4): {pprint(population_with_fitness)}"

    for ind in population_with_fitness:
        logger.info(f"Individual {ind.id} has fitness {ind.fitness_value}")

    assert max((ind.fitness_value[AVG_COHERENCE_SCORE] for ind in population_with_fitness)) > 0, \
        "At least one fitness should be more than zero"

    logger.info("Test run succeded")
    click.echo('Initialized the database')


@cli.command()
@click.option('--timeout', type=float, default=25.0)
@click.option('--expid', type=int)
def run_and_log(timeout: float, expid: Optional[str] = None):
    best_ind = population[0]
    if expid:
        best_ind.exp_id = expid
    logger.info(f"Individual: {best_ind}")
    ind = log_best_solution(individual=best_ind, wait_for_result_timeout=timeout, alg_args="--arg 1 --arg 2")
    logger.info(f"Logged the best solution. Obtained fitness is {ind.fitness_value[AVG_COHERENCE_SCORE]}")


if __name__ == '__main__':
    cli()
