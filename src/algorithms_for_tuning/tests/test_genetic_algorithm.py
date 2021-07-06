from src.algorithms_for_tuning.genetic_algorithm.mutation import positioning_mutation
from src.algorithms_for_tuning.genetic_algorithm.crossover import crossover_pmx
from src.algorithms_for_tuning.genetic_algorithm.selection import selection_rank_based

import pytest
from dataclasses import dataclass
from dataclasses_json import dataclass_json
from typing import List, Optional
import random


@dataclass_json
@dataclass
class IndividualDTO:
    id: str
    params: List[object]
    fitness_name: str = "default"
    dataset: str = "default"
    force_dataset_settings_checkout: bool = False
    fitness_value: Optional[float] = None


@pytest.fixture
def generate_individuals(n=10):
    list_of_individuals = []
    for i in range(n):
        list_of_individuals.append(
            IndividualDTO(id=n,
                          params=[],
                          fitness_value=random.random()
                          )
        )


@pytest.mark.parametrize(
    "individ", "elem_mutation_prob",
    [],
    []
)
def test_positioning_mutation(individ, elem_mutation_prob=0.1):
    pass


def test_crossover_pmx(parent_1, parent_2):
    pass


def test_selection_rank_based(population, best_proc, children_num):
    pass
