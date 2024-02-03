import random
from typing import List, Union

from pydantic import BaseModel


class IntRangeDistribution(BaseModel):
    low: int
    high: int

    def create_value(self) -> int:
        return random.randint(self.low, self.high)

    def clip(self, value) -> int:
        value = int(value)
        return int(max(self.low, min(self.high, value)))


class FloatRangeDistribution(BaseModel):
    low: float
    high: float

    def create_value(self) -> float:
        return random.uniform(self.low, self.high)

    def clip(self, value) -> float:
        return max(self.low, min(self.high, value))


class Param(BaseModel):
    """
    Single parameter of a stage.
    Distribution can be unavailable after serialisation.
    """
    name: str
    distribution: Union[IntRangeDistribution, FloatRangeDistribution]

    def create_value(self):
        if self.distribution is None:
            raise ValueError("Distribution is unavailable. One must restore the distribution after serialisation.")
        return self.distribution.create_value()


class StageType(BaseModel):
    """
    Stage template that defines params for a stage.
    See create_stage generator.
    """
    name: str
    params: List[Param]


class Stage(BaseModel):
    """
    Stage instance with parameter values.
    """
    stage_type: StageType
    values: List

    def __init__(self, **data):
        """
        :param stage_type: back reference to the template for parameter mutation
        :param values: instance's parameter values
        """
        super().__init__(**data)
        if len(self.stage_type.params) != len(self.values):
            raise ValueError("Number of values does not match number of parameters.")

    def __str__(self):
        return f"{self.stage_type.name}{self.values}"

    def clip_values(self):
        self.values = [param.distribution.clip(value) for param, value in zip(self.stage_type.params, self.values)]


def create_stage(stage_type: StageType) -> Stage:
    return Stage(stage_type=stage_type, values=[param.create_value() for param in stage_type.params])


class Pipeline(BaseModel):
    """
    List of stages that can be mutated.
    """
    stages: List[Stage]
    required_params: Stage

    def __str__(self):
        return f'{str(self.required_params)} {" ".join(map(str, self.stages))}'

    def random_stage_index(self, with_last: bool = False):
        last = len(self.stages)
        if with_last:
            last += 1
        return random.randint(0, last - 1)

    def __lt__(self, other):
        # important for sort method usage
        return len(self.stages) < len(other.stages)
