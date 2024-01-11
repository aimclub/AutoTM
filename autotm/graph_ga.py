from typing import Tuple, Callable

from autotm.algorithms_for_tuning.genetic_algorithm.crossover import crossover_one_point
from autotm.pipeline import *


# util methods

def random_action(probability: float):
    """
    Bernoulli's distribution.
    :param probability: True probability
    :return: True with given probability, or False otherwise
    """
    return random.random() < probability


def create_or_default(create: Callable, probability: float, default_value):
    """
    Create new value with given probability or use the default one otherwise.
    """
    return create() if random_action(probability) else default_value


def replace_values(original: List, new: List):
    """
    Replace values in a list with values from another list.
    """
    original.clear()
    for e in new:
        original.append(e)


def sort_by_values(keys: List, values: List[float]):
    """
    Sort given keys by associated values.
    """
    data = list(zip(values, keys))
    data.sort(reverse=True)
    sorted_values, sorted_keys = zip(*data)

    replace_values(keys, sorted_keys)
    replace_values(values, sorted_values)


def sort_by_value(keys: List, value_gen: Callable) -> List[float]:
    """
    Sort given keys by values generated via passed lambda and return the values.
    """
    values = list(map(value_gen, keys))
    sort_by_values(keys, values)
    return values


def remove_all(elements, elements_to_remove):
    for e in elements_to_remove:
        if e in elements:
            elements.remove(e)


def single_point_crossover(list1: List, list2: List) -> Tuple[List, List]:
    while True:
        i1 = random.randint(0, len(list1))
        i2 = random.randint(0, len(list2))
        if (i1 == 0 and i2 == 0) or (i1 == len(list1) and i2 == len(list2)):
            continue
        break

    return list1[:i1] + list2[i2:], list2[:i2] + list1[i1:]


def crossover_stage(stage1: Stage, stage2: Stage) -> Tuple[Stage, Stage]:
    if stage1.stage_type != stage2.stage_type:
        raise ValueError("Cannot crossover stages with different types")
    values1, values2 = crossover_one_point(stage1.values, stage2.values, elem_cross_prob=0.2)
    return Stage(stage_type=stage1.stage_type, values=values1), Stage(stage_type=stage2.stage_type, values=values2)


def crossover_pipelines(pipeline1: Pipeline, pipeline2: Pipeline) -> List[Pipeline]:
    """
    Single-point crossover of two pipelines.
    Returns non-empty pipelines after crossover.
    """

    stages1, stages2 = single_point_crossover(pipeline1.stages, pipeline2.stages)
    values1, values2 = crossover_stage(pipeline1.required_params, pipeline2.required_params)

    child1 = Pipeline(stages=stages1, required_params=values1)
    child2 = Pipeline(stages=stages2, required_params=values2)

    return [child for child in [child1, child2] if len(child.stages) > 0]


def mutate_stage(stage: Stage, mutation_probability: float) -> Stage:
    """
    Mutate parameters of a single stage.
    :param stage: stage to mutate
    :param mutation_probability: Probability of single parameter mutation
    :return: a mutated copy of the stage
    """
    mutated_values = [create_or_default(param.create_value, mutation_probability, value) for param, value
                      in zip(stage.stage_type.params, stage.values)]
    return Stage(stage_type=stage.stage_type, values=mutated_values)


def mutate_required_params(pipeline: Pipeline, mutation_probability: float) -> Pipeline:
    """
    Mutate required parameters of the pipeline.
    :param pipeline: pipeline to mutate
    :param mutation_probability: Probability of single parameter mutation
    :return: a mutated copy of the pipeline
    """
    mutated_params = mutate_stage(pipeline.required_params, mutation_probability)
    return Pipeline(stages=pipeline.stages, required_params=mutated_params)


def mutate_stages(pipeline: Pipeline, mutation_probability: float) -> Pipeline:
    """
    Mutate parameters of a single stage from the pipeline.
    :param pipeline: pipeline to mutate
    :param mutation_probability: Probability of single parameter mutation
    :return: a mutated copy of the pipeline
    """
    i = pipeline.random_stage_index()
    mutated_stage = mutate_stage(pipeline.stages[i], mutation_probability)
    return Pipeline(stages=pipeline.stages[:i] + [mutated_stage] + pipeline.stages[i + 1:],
                    required_params=pipeline.required_params)


def remove_stage(pipeline: Pipeline):
    """
    Remove random stage from the pipeline.
    Can be applied only to pipelines with size more than 1.
    :param pipeline: pipeline to mutate
    :return: a mutated copy of the pipeline
    """
    if len(pipeline.stages) <= 1:
        return pipeline
    deletion_index = pipeline.random_stage_index()
    return Pipeline(stages=pipeline.stages[:deletion_index] + pipeline.stages[deletion_index + 1:],
                    required_params=pipeline.required_params)


def add_stage(pipeline: Pipeline, stage_types: List[StageType]):
    """
    Add random stage to the pipeline.
    :param pipeline: pipeline to mutate
    :param stage_types: stage types that can be used as a new stage
    :return: a mutated copy of the pipeline
    """
    stage = create_stage(random.choice(stage_types))
    insertion_index = pipeline.random_stage_index(with_last=True)
    return Pipeline(stages=pipeline.stages[:insertion_index] + [stage] + pipeline.stages[insertion_index:],
                    required_params=pipeline.required_params)


def swap_stages(pipeline: Pipeline):
    """
    Swap to random stages in the pipeline.
    Can be applied only to pipelines with size more than 1.
    :param pipeline: pipeline to mutate
    :return: a mutated copy of the pipeline
    """
    if len(pipeline.stages) <= 1:
        return pipeline
    i1 = pipeline.random_stage_index()
    i2 = pipeline.random_stage_index()
    while i1 == i2:
        i2 = pipeline.random_stage_index()

    stages = [stage for stage in pipeline.stages]
    stages[i1], stages[i2] = stages[i2], stages[i1]
    return Pipeline(stages=stages, required_params=pipeline.required_params)


def mutate_pipeline(pipeline: Pipeline, stage_types: List[StageType], stage_mutation_probability: float) -> Pipeline:
    """
    Apply one mutation to the pipeline.
    :param pipeline: pipeline to mutate
    :param stage_types: stage types that can be used as a new stage
    :param stage_mutation_probability: Probability of single parameter mutation
    :return: a mutated copy of the pipeline
    """
    mutations = [lambda p: add_stage(p, stage_types),
                 lambda p: mutate_stages(p, stage_mutation_probability),
                 ]
    weights = [1, 7]
    if len(pipeline.stages) > 1:
        mutations.append(remove_stage)
        weights.append(1)
        mutations.append(swap_stages)
        weights.append(1)
    if len(pipeline.required_params.stage_type.params) > 0:
        mutations.append(lambda p: mutate_required_params(p, stage_mutation_probability))
        weights.append(3)
    mutation = random.choices(mutations, weights=weights)[0]
    return mutation(pipeline)


def create_pipeline(stage_types: List[StageType],
                    pipeline_size_distribution: Callable[[], int],
                    required_stage_type: StageType) -> Pipeline:
    size = pipeline_size_distribution()
    return Pipeline(stages=list(map(create_stage, random.choices(stage_types, k=size))),
                    required_params=create_stage(required_stage_type))


def create_initial_population(stage_types: List[StageType],
                              population_size: int,
                              pipeline_size_distribution: Callable[[], int],
                              required_stage_type: StageType) -> List[Pipeline]:
    return [create_pipeline(stage_types, pipeline_size_distribution, required_stage_type)
            for _ in range(population_size)]


def tournament_selection(population: List[Pipeline], tournament_size: int) -> List[Pipeline]:
    tournament_contestants = random.sample(range(len(population)), tournament_size)
    tournament_contestants.sort()
    w1, w2 = tournament_contestants[:2]
    return [population[w1], population[w2]]


def genetic_algorithm(population: List[Pipeline],
                      generations: int,
                      stage_types: List[StageType],
                      fitness: Callable[[Pipeline], float],
                      crossover_probability: float,
                      mutation_probability: float,
                      stage_mutation_probability: float,
                      generation_callback: Callable[[List[Pipeline], List[float]], None] = None,
                      tournament_size=5) -> Tuple[List[Pipeline], List[float]]:
    population_size = len(population)
    fitness_values = sort_by_value(population, fitness)

    for generation in range(generations):
        new_population = []

        while len(new_population) < population_size:
            parent1, parent2 = tournament_selection(population, tournament_size)
            children = create_or_default(lambda: crossover_pipelines(parent1, parent2),
                                         crossover_probability,
                                         [parent1, parent2])
            children = [create_or_default(lambda: mutate_pipeline(child, stage_types, stage_mutation_probability),
                                          mutation_probability,
                                          child) for child in children]
            remove_all(children, [parent1, parent2])
            new_population += children

        new_fitness_values = sort_by_value(new_population, fitness)

        population += new_population
        fitness_values += new_fitness_values

        sort_by_values(population, fitness_values)

        if generation_callback is not None:
            generation_callback(population, fitness_values)

        population = population[:population_size]
        fitness_values = fitness_values[:population_size]

    return population, fitness_values
