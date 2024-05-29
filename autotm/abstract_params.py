from abc import ABC, abstractmethod

from typing import List


class AbstractParams(ABC):
    @property
    @abstractmethod
    def basic_topics(self) -> int:
        ...

    @property
    @abstractmethod
    def mutation_probability(self):
        ...

    @abstractmethod
    def make_params_dict(self):
        ...

    @abstractmethod
    def run_train(self, model: "TopicModel"):
        ...

    @abstractmethod
    def validate_params(self) -> bool:
        ...

    @abstractmethod
    def crossover(self, parent2: "AbstractParams", **kwargs) -> List["AbstractParams"]:
        ...

    @abstractmethod
    def mutate(self, **kwargs) -> "AbstractParams":
        ...

    @abstractmethod
    def to_vector(self) -> List[float]:
        ...

