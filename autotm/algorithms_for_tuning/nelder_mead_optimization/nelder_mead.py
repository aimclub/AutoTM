import numpy as np
from scipy.optimize import minimize

from autotm.fitness.tm import FitnessCalculatorWrapper


class NelderMeadOptimization:
    def __init__(
        self,
        dataset,
        data_path,
        exp_id,
        topic_count,
        train_option,
        low_decor=0,
        high_decor=1e5,
        low_n=0,
        high_n=30,
        low_back=0,
        high_back=5,
        low_spb=0,
        high_spb=1e2,
        low_spm=-1e-3,
        high_spm=1e2,
        low_sp_phi=-1e3,
        high_sp_phi=1e3,
        low_prob=0,
        high_prob=1,
    ):
        self.dataset = (dataset,)
        self.data_path = data_path
        self.exp_id = exp_id
        self.topic_count = topic_count
        self.train_option = train_option
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

    def initialize_params(self):
        val_decor = np.random.uniform(low=self.low_decor, high=self.high_decor, size=1)[
            0
        ]
        var_n = np.random.randint(low=self.low_n, high=self.high_n, size=4)
        var_back = np.random.randint(low=self.low_back, high=self.high_back, size=1)[0]
        var_sm = np.random.uniform(low=self.low_spb, high=self.high_spb, size=2)
        var_sp = np.random.uniform(low=self.low_sp_phi, high=self.high_sp_phi, size=4)
        val_decor_2 = np.random.uniform(
            low=self.low_decor, high=self.high_decor, size=1
        )[0]
        params = [
            val_decor,
            var_n[0],
            var_sm[0],
            var_sm[1],
            var_n[1],
            var_sp[0],
            var_sp[1],
            var_n[2],
            var_sp[2],
            var_sp[3],
            var_n[3],
            var_back,
            val_decor_2,
        ]
        params = [float(i) for i in params]
        return params

    def run_algorithm(self, num_iterations: int = 400, ini_point: list = None):
        fitness_calculator = FitnessCalculatorWrapper(
            self.dataset, self.data_path, self.topic_count, self.train_option
        )

        if ini_point is None:
            initial_point = self.initialize_params()
        else:
            assert len(ini_point) == 13
            print(ini_point)  # TODO: remove this
            ini_point = [float(i) for i in ini_point]
            initial_point = ini_point

        res = minimize(
            fitness_calculator.run,
            initial_point,
            bounds=[
                (self.low_decor, self.high_decor),
                (self.low_n, self.high_n),
                (self.low_spb, self.high_spb),
                (self.low_spb, self.high_spb),
                (self.low_n, self.high_n),
                (self.low_sp_phi, self.high_sp_phi),
                (self.low_sp_phi, self.high_sp_phi),
                (self.low_n, self.high_n),
                (self.low_sp_phi, self.high_sp_phi),
                (self.low_sp_phi, self.high_sp_phi),
                (self.low_n, self.high_n),
                (self.low_back, self.high_back),
                (self.low_decor, self.high_decor),
            ],
            method="Nelder-Mead",
            options={"return_all": True, "maxiter": num_iterations},
        )

        return res
