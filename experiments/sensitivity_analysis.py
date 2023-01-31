from SALib.analyze import sobol
import numpy as np


def init_individ(self, base_model=False):
    val_decor = np.random.uniform(low=self.low_decor, high=self.high_decor, size=1)[0]
    var_n = np.random.randint(low=self.low_n, high=self.high_n, size=4)
    var_back = np.random.randint(low=self.low_back, high=self.high_back, size=1)[0]
    var_sm = np.random.uniform(low=self.low_spb, high=self.high_spb, size=2)
    var_sp = np.random.uniform(low=self.low_sp_phi, high=self.high_sp_phi, size=4)
    ext_mutation_prob = np.random.uniform(low=self.low_prob, high=self.high_prob, size=1)[0]
    ext_elem_mutation_prob = np.random.uniform(low=self.low_prob, high=self.high_prob, size=1)[0]
    ext_mutation_selector = np.random.uniform(low=self.low_prob, high=self.high_prob, size=1)[0]
    val_decor_2 = np.random.uniform(low=self.low_decor, high=self.high_decor, size=1)[0]
    params = [
        val_decor, var_n[0],
        var_sm[0], var_sm[1], var_n[1],
        var_sp[0], var_sp[1], var_n[2],
        var_sp[2], var_sp[3], var_n[3],
        var_back,
        ext_mutation_prob, ext_elem_mutation_prob, ext_mutation_selector,
        val_decor_2
    ]
    if base_model:
        for i in [0, 4, 7, 10, 11, 15]:
            params[i] = 0
    params = [float(i) for i in params]
    return params

def run_sensitivity_analysis():
    problem = {
        'num_var': 13,
        'names': ['decor', 'back'],
        'bounds': [
            [low=self.low_decor, high=self.high_decor],

        ]],
    }