from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List

from kube_fitness.metrics import AVG_COHERENCE_SCORE
from kube_fitness.schemas import IndividualDTO

SPARSITY_PHI = 'sparsity_phi'
SPARSITY_THETA = 'sparsity_theta'


class Individual(ABC):
    @property
    @abstractmethod
    def dto(self) -> IndividualDTO:
        ...

    @property
    @abstractmethod
    def fitness_value(self) -> float:
        ...

    @property
    @abstractmethod
    def params(self) -> List:
        ...


class RegularFitnessIndividual(Individual):
    def __init__(self, dto: IndividualDTO):
        self._dto = dto

    @property
    def dto(self) -> IndividualDTO:
        return self._dto

    @property
    def fitness_value_prev(self) -> float:
        return self.dto.fitness_value[AVG_COHERENCE_SCORE]

    @property
    def fitness_value(self) -> float:
        alpha = 0.8
        if 0.2 <= self.dto.fitness_value[SPARSITY_THETA] <= 0.8:
            alpha = 1
        return alpha * self.dto.fitness_value[AVG_COHERENCE_SCORE]

    @property
    def params(self) -> List:
        return self.dto.params


def make_individual(dto: IndividualDTO) -> Individual:
    return RegularFitnessIndividual(dto=dto)
