from dataclasses import dataclass
from typing import List, Optional
import random
import time
from tqdm import tqdm


@dataclass
class IndividualDTO:
    id: str
    params: List[object]
    fitness_name: str = "default"
    dataset: str = "default"
    force_dataset_settings_checkout: bool = False
    fitness_value: Optional[float] = None


def parallel_fitness(population: List[IndividualDTO],
                     use_tqdm: bool = False,
                     tqdm_check_period: int = 2) -> List[IndividualDTO]:
    ids = [ind.id for ind in population]
    assert len(set(ids)) == len(population), \
        f"There are individuals with duplicate ids: {ids}"

    results = []

    for individ in population:
        individ.fitness_value = random.random()
        results.append(individ.fitness_value)

    # restoring the order in the resulting population according to the initial population
    results_by_id = {ind.id: ind for ind in (r for r in results)}
    return [results_by_id[ind.id] for ind in population]

    # return [results_by_id[ind.id] for ind in population]
