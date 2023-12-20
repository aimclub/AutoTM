from abc import ABC, abstractmethod
import os
import pickle
from typing import List
import numpy as np
import pandas as pd

from autotm.utils import AVG_COHERENCE_SCORE
from autotm.schemas import IndividualDTO

SPARSITY_PHI = "sparsity_phi"
SPARSITY_THETA = "sparsity_theta"
SWITCHP_SCORE = "switchP"
DF_NAMES = {"20ng": 0, "lentaru": 1, "amazon_food": 2}

METRICS_COLS = [
    "avg_coherence_score",
    "perplexityScore",
    "backgroundTokensRatioScore",
    "avg_switchp",
    "coherence_10",
    "coherence_15",
    "coherence_20",
    "coherence_25",
    "coherence_30",
    "coherence_35",
    "coherence_40",
    "coherence_45",
    "coherence_50",
    "coherence_55",
    "contrast",
    "purity",
    "kernelSize",
    "sparsity_phi",
    "sparsity_theta",
    "topic_significance_uni",
    "topic_significance_vacuous",
    "topic_significance_back",
    "npmi_15",
    "npmi_25",
    "npmi_50",
]

PATH_TO_LEARNED_SCORING = "./scoring_func"


class Individual(ABC):
    id: str

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


class LearnedModel:
    def __init__(self, save_path, dataset_name):
        dataset_id = DF_NAMES[dataset_name]
        general_save_path = os.path.join(save_path, "general")
        native_save_path = os.path.join(save_path, "native")
        with open(
            os.path.join(general_save_path, f"general_automl_{dataset_id}.pickle"), "rb"
        ) as f:
            self.general_model = pickle.load(f)
        self.native_model = []
        for i in range(5):
            with open(
                os.path.join(
                    native_save_path, f"native_automl_{dataset_id}_fold_{i}.pickle"
                ),
                "rb",
            ) as f:
                self.native_model.append(pickle.load(f))

    def general_predict(self, df: pd.DataFrame):
        y = self.general_model.predict(df[METRICS_COLS])
        return y

    def native_predict(self, df: pd.DataFrame):
        y = []
        for k, nm in enumerate(self.native_model):
            y.append(nm.predict(df[METRICS_COLS]))
        y = np.array(y)
        return np.mean(y, axis=0)


class LearnedScorerFitnessIndividual(BaseIndividual):
    @property
    def fitness_value(self) -> float:
        # dataset_name = self.dto.dataset  # TODO: check namings
        # m = LearnedModel(save_path=PATH_TO_LEARNED_SCORING, dataset_name=dataset_name)
        # TODO: predict from metrics df
        raise NotImplementedError()


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
        # if SWITCHP_SCORE in self.dto.fitness_value:
        #     return alpha * (self.dto.fitness_value[AVG_COHERENCE_SCORE] + self.dto.fitness_value[SWITCHP_SCORE])
        # else:
        #     return alpha * self.dto.fitness_value[AVG_COHERENCE_SCORE]
        return alpha * self.dto.fitness_value[AVG_COHERENCE_SCORE]


def make_individual(dto: IndividualDTO) -> Individual:
    # TODO: choose fitness by ENV var
    return RegularFitnessIndividual(dto=dto)
    # return SparsityScalerBasedFitnessIndividual(dto=dto)
