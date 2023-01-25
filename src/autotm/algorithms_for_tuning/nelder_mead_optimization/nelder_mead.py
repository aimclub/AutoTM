from scipy.optimize import minimize
from typing import Union
from autotm.fitness import


class NelderMeadOptimization:

    def __init__(self,
                 low_decor=0, high_decor=1e5,
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

    def run_algorithm(self, dataset: str,
                      data_path: str,
                      exp_id: Union[int, str],
                      topic_count: int,
                      num_individuals: int = 11,
                      num_iterations: int = 400):
        minimize()
        pass
