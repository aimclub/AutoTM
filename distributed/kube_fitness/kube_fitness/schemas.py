import copy
from typing import List, Optional

from pydantic import BaseModel

from kube_fitness.metrics import MetricsScores

PARAM_NAMES = [
    'val_decor', 'var_n_0', 'var_sm_0', 'var_sm_1', 'var_n_1',
    'var_sp_0', 'var_sp_1', 'var_n_2',
    'var_sp_2', 'var_sp_3', "var_n_3",
    'var_n_4',
    'ext_mutation_prob', 'ext_elem_mutation_prob', 'ext_mutation_selector',
    'val_decor_2'
]


# @dataclass_json
# @dataclass
class IndividualDTO(BaseModel):
    id: str
    params: List[object]
    fitness_name: str = "default"
    dataset: str = "default"
    force_dataset_settings_checkout: bool = False
    fitness_value: MetricsScores = None
    exp_id: Optional[int] = None
    alg_id: Optional[str] = None
    tag: Optional[str] = None
    iteration_id: int = 0
    topic_count: Optional[int] = None
    train_option: str = 'offline'

    def make_params_dict(self):
        if len(self.params) > len(PARAM_NAMES):
            len_diff = len(self.params) - len(PARAM_NAMES)
            param_names = copy.deepcopy(PARAM_NAMES) + [f"unknown_param_#{i}" for i in range(len_diff)]
        else:
            param_names = PARAM_NAMES

        return {name: p_val for name, p_val in zip(param_names, self.params)}
