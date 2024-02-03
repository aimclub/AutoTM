from typing import Optional
from typing import Union

from pydantic import BaseModel

from autotm.params import PipelineParams, FixedListParams
from autotm.utils import MetricsScores

AnyParams = Union[FixedListParams, PipelineParams]


class IndividualDTO(BaseModel):
    id: str
    data_path: str
    params: AnyParams
    fitness_name: str = "default"
    dataset: str = "default"
    force_dataset_settings_checkout: bool = False
    fitness_value: Optional[MetricsScores] = None
    exp_id: Optional[int] = None
    alg_id: Optional[str] = None
    tag: Optional[str] = None
    iteration_id: int = 0
    topic_count: Optional[int] = None
    train_option: str = "offline"

    class Config:
        arbitrary_types_allowed = True

    def make_params_dict(self):
        return self.params.make_params_dict()

