from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List

from kube_fitness.metrics import AVG_COHERENCE_SCORE
from kube_fitness.schemas import IndividualDTO


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
    def fitness_value(self) -> float:
        return self.dto.fitness_value[AVG_COHERENCE_SCORE]

    @property
    def params(self) -> List:
        return self.dto.params


def make_individual(dto: IndividualDTO) -> Individual:
    return RegularFitnessIndividual(dto=dto)
