# multistage bag of regularizers approach

import numpy as np
import random


class ModelStorage:
    def __init__(self):
        self.stage_1_components = {}  # {config_id: id}
        self.stage_1_hyperp = {}  # {config_id: [[params1, params2]]}

    def model_search(self, model):
        raise NotImplementedError
        # for model.components


class GA:
    def __init__(self, dataset, max_stages=5):  # max_stage_len
        self.max_stages = max_stages  # amount of unique regularizers
        self.dataset = dataset
        self.bag_of_regularizers = ['decor_S', 'decor_B', 'S_phi_B', 'S_phi_S',
                                    'S_theta_B', 'S_theta_S']  # add separate smooth and sparsity

        self.initial_element_stage_probability = 0.5
        self.positioning_matrix = np.full((len(self.bag_of_regularizers), self.max_stages - 1), 0.5)
        self.set_regularizer_limits()

    def set_regularizer_limits(self, low_decor=0, high_decor=1e5,
                               low_n=0, high_n=30,
                               low_back=0, high_back=5,
                               low_spb=0, high_spb=1e2,
                               low_spm=-1e-3, high_spm=1e2,
                               low_sp_phi=-1e3, high_sp_phi=1e3,
                               low_prob=0, high_prob=1):
        self.high_decor = high_decor
        self.low_decor = low_decor
        self.low_n = low_n
        self.high_n = high_n
        self.low_back = low_back
        self.high_back = high_back
        self.high_spb = high_spb
        self.low_spb = low_spb
        self.low_spm = low_spm
        self.high_spm = high_spm
        self.low_sp_phi = low_sp_phi
        self.high_sp_phi = high_sp_phi
        self.low_prob = low_prob
        self.high_prob = high_prob

    def _init_param(self, param_type):
        if param_type == 'decor_S' or param_type == 'decor_B':
            return np.random.uniform(low=self.low_decor, high=self.high_decor, size=1)[0]
        elif param_type == 'S_phi_B' or 'S_theta_B':
            return np.random.uniform(low=self.low_spb, high=self.high_spb, size=1)[0]
        elif param_type == 'S_phi_S' or 'S_theta_S':
            return np.random.uniform(low=self.low_sp_phi, high=self.high_sp_phi, size=1)[0]
        elif param_type == 'n':
            return np.random.randint(low=self.low_n, high=self.high_n, size=1)[0]
        elif param_type == 'B':
            return np.random.randint(low=self.low_back, high=self.high_back, size=1)[0]

    def _create_stage(self, stage_num):
        for ix, elem in enumerate(self.bag_of_regularizers):
            self.positioning_matrix[stage_num][2][stage_num - 1]
        raise NotImplementedError

    def init_individ(self):
        number_of_stages = np.random.randint(low=1, high=self.max_stages, size=1)[0]
        for i in range(number_of_stages):
            regularizers = self._create_stage(i)


        if random.random() < self.initial_element_stage_probability:

        for i in range(self.max_stages):
            print()

        raise NotImplementedError

    def ffff(self):
        raise NotImplementedError
