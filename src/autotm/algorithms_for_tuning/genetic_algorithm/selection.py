import operator
import numpy as np
import random


# TODO: roulette wheel selection, stochastic universal sampling and tournament selection


def yield_matching_pairs(pairs, population):
    # print('Number of pairs: {}'.format(pairs))
    population.sort(key=operator.attrgetter("fitness_value"))
    population_pairs_pool = []

    while len(population_pairs_pool) < pairs:
        chosen = []
        idx = 0
        selection_probability = random.random()
        for ix, individ in enumerate(population):
            if selection_probability <= individ._prob:
                idx = ix
                chosen.append(individ)
                break

        selection_probability = random.random()
        for k, individ in enumerate(population):
            if k != idx:
                if selection_probability <= individ._prob:
                    elems = frozenset((idx, k))
                    if (len(population_pairs_pool) == 0) or (
                        (len(population_pairs_pool) > 0)
                        and (elems not in population_pairs_pool)
                    ):
                        chosen.append(individ)
                        population_pairs_pool.append(elems)
                        break
            else:
                continue
        if len(chosen) == 1:
            selection_idx = np.random.choice(
                [m for m in [i for i in range(len(population))] if m != idx]
            )
            chosen.append(population[selection_idx])
        if len(chosen) == 0:
            yield None, None
        else:
            yield chosen[0], chosen[1]


def selection_fitness_prop(population, best_proc, children_num):
    all_fitness = []

    for individ in population:
        all_fitness.append(individ.fitness_value)
    fitness_std = np.std(all_fitness)
    fitness_mean = np.mean(all_fitness)
    cumsum_fitness = 0
    # adjust probabilities with sigma scaling
    c = 2
    for individ in population:
        updated_individ_fitness = np.max(
            [(individ.fitness_value - (fitness_mean - c * fitness_std)), 0]
        )
        cumsum_fitness += updated_individ_fitness
        individ._prob = updated_individ_fitness / cumsum_fitness
    if children_num == 2:
        return yield_matching_pairs(
            round((len(population) * (1 - best_proc)) // 2), population
        )
    else:
        return yield_matching_pairs(
            round((len(population) * (1 - best_proc))), population
        )


def selection_rank_based(population, best_proc, children_num):
    population.sort(key=operator.attrgetter("fitness_value"))
    for ix, individ in enumerate(population):
        individ._prob = 2 * (ix + 1) / (len(population) * (len(population) - 1))
    if children_num == 2:
        # new population size
        return yield_matching_pairs(
            round((len(population) * (1 - best_proc))), population
        )
    else:
        return yield_matching_pairs(
            round((len(population) * (1 - best_proc))), population
        )


def stochastic_universal_sampling():
    raise NotImplementedError


def selection(selection_type="fitness_prop"):
    if selection_type == "fitness_prop":
        return selection_fitness_prop
    if selection_type == "rank_based":
        return selection_rank_based
