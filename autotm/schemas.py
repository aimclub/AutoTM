import copy
import json
from typing import List, Optional
from pydantic import BaseModel

from autotm.utils import MetricsScores

PARAM_NAMES = [
    "val_decor",
    "var_n_0",
    "var_sm_0",
    "var_sm_1",
    "var_n_1",
    "var_sp_0",
    "var_sp_1",
    "var_n_2",
    "var_sp_2",
    "var_sp_3",
    "var_n_3",
    "var_n_4",
    "ext_mutation_prob",
    "ext_elem_mutation_prob",
    "ext_mutation_selector",
    "val_decor_2",
]


class IndividualDTO(BaseModel):
    id: str
    data_path: str
    params: List[object]
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
        if len(self.params) > len(PARAM_NAMES):
            len_diff = len(self.params) - len(PARAM_NAMES)
            param_names = copy.deepcopy(PARAM_NAMES) + [
                f"unknown_param_#{i}" for i in range(len_diff)
            ]
        else:
            param_names = PARAM_NAMES

        return {name: p_val for name, p_val in zip(param_names, self.params)}


def fitness_to_json(obj: IndividualDTO):
    return json.dumps(
        {
            "id": obj.id,
            "data_path": obj.data_path,
            "params": obj.params,
            "exp_id": obj.exp_id,
            "alg_id": obj.alg_id,
            "iteration_id": obj.iteration_id,
            "tag": obj.tag,
            "fitness_value": obj.fitness_value,
            "fitness_name": obj.fitness_name,
            "dataset": obj.dataset,
            "topic_count": obj.topic_count,
            "train_option": obj.train_option,
            "force_dataset_settings_checkout": obj.force_dataset_settings_checkout,
        }
    )


def fitness_from_json(obj):
    obj = json.loads(obj)
    return IndividualDTO(
        id=obj["id"],
        data_path=obj["data_path"],
        params=obj["params"],
        exp_id=int(obj["exp_id"]) if obj["exp_id"] else None,
        alg_id=obj["alg_id"] if obj["alg_id"] else None,
        iteration_id=obj["iteration_id"],
        tag=obj["tag"],
        fitness_value=obj["fitness_value"],
        fitness_name=obj["fitness_name"],
        dataset=obj["dataset"],
        topic_count=obj["topic_count"],
        train_option=obj["train_option"],
        force_dataset_settings_checkout=obj["force_dataset_settings_checkout"],
    )
