from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List

from kube_fitness.metrics import AVG_COHERENCE_SCORE
from kube_fitness.schemas import IndividualDTO

SPARSITY_PHI = 'sparsity_phi'
SPARSITY_THETA = 'sparsity_theta'
SWITCHP_SCORE = 'switchP'

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


class BaseIndividual(Individual, ABC):
    def __init__(self, dto: IndividualDTO):
        self._dto = dto

    @property
    def dto(self) -> IndividualDTO:
        return self._dto

    @property
    def params(self) -> List:
        return self.dto.params


class RegularFitnessIndividual(BaseIndividual):
    @property
    def fitness_value(self) -> float:
        return self.dto.fitness_value[AVG_COHERENCE_SCORE]


class SparsityScalerBasedFitnessIndividual(BaseIndividual):
    @property
    def fitness_value(self) -> float:
        # it is a handling of the situation when a fitness-worker wasn't able to correctly calculate this indvidual
        # due to some error in the proceess
        # and thus the fitness value doesn't have any metrics except dummy AVG_COHERENCE_SCORE equal to zero
        if self.dto.fitness_value[AVG_COHERENCE_SCORE] < 0.00000001:
            return 0.0

        alpha = 0.7
        if 0.2 <= self.dto.fitness_value[SPARSITY_THETA] <= 0.8:
            alpha = 1
        if SWITCHP_SCORE in self.dto.fitness_value:
            return alpha * (self.dto.fitness_value[AVG_COHERENCE_SCORE] + self.dto.fitness_value[SWITCHP_SCORE])
        else:
            return alpha * self.dto.fitness_value[AVG_COHERENCE_SCORE]
        # return alpha * self.dto.fitness_value[AVG_COHERENCE_SCORE]

def make_individual(dto: IndividualDTO) -> Individual:
    # TODO: choose fitness by ENV var
    # return RegularFitnessIndividual(dto=dto)
    return SparsityScalerBasedFitnessIndividual(dto=dto)
