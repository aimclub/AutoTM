import logging

import numpy as np
from sklearn.ensemble import RandomForestRegressor, BaggingRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, Matern, WhiteKernel, ExpSineSquared, RationalQuadratic
from sklearn.metrics import mean_squared_error
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor

logger = logging.getLogger("GA_algo")


# TODO: Add fitness type
def set_surrogate_fitness(value, fitness_type="avg_coherence_score"):
    npmis = {
        "npmi_50": None,
        "npmi_15": None,
        "npmi_25": None,
        "npmi_50_list": None,
    }
    scores_dict = {
        fitness_type: value,
        "perplexityScore": None,
        "backgroundTokensRatioScore": None,
        "contrast": None,
        "purity": None,
        "kernelSize": None,
        "npmi_50_list": [None],  # npmi_values_50_list,
        "npmi_50": None,
        "sparsity_phi": None,
        "sparsity_theta": None,
        "topic_significance_uni": None,
        "topic_significance_vacuous": None,
        "topic_significance_back": None,
        "switchP_list": [None],
        "switchP": None,
        "all_topics": None,
        # **coherence_scores,
        **npmis,
    }
    return scores_dict


class Surrogate:
    def __init__(self, surrogate_name, **kwargs):
        self.name = surrogate_name
        self.kwargs = kwargs
        self.surrogate = None
        self.br_n_estimators = None
        self.br_n_jobs = None
        self.gpr_kernel = None

    def create(self):
        kernel = self.kwargs["gpr_kernel"]
        del self.kwargs["gpr_kernel"]
        gpr_alpha = self.kwargs["gpr_alpha"]
        del self.kwargs["gpr_alpha"]
        normalize_y = self.kwargs["normalize_y"]
        del self.kwargs["normalize_y"]

        if self.name == "random-forest-regressor":
            self.surrogate = RandomForestRegressor(**self.kwargs)
        elif self.name == "mlp-regressor":
            if not self.br_n_estimators:
                self.br_n_estimators = self.kwargs["br_n_estimators"]
                del self.kwargs["br_n_estimators"]
                self.br_n_jobs = self.kwargs["n_jobs"]
                del self.kwargs["n_jobs"]
                self.kwargs["alpha"] = self.kwargs["mlp_alpha"]
                del self.kwargs["mlp_alpha"]
            self.surrogate = BaggingRegressor(
                base_estimator=MLPRegressor(**self.kwargs),
                n_estimators=self.br_n_estimators,
                n_jobs=self.br_n_jobs,
            )
        elif self.name == "GPR":  # tune ??
            if not self.gpr_kernel:
                if kernel == "RBF":
                    self.gpr_kernel = 1.0 * RBF(1.0)
                elif kernel == "RBFwithConstant":
                    self.gpr_kernel = 1.0 * RBF(1.0) + ConstantKernel()
                elif kernel == "Matern":
                    self.gpr_kernel = 1.0 * Matern(1.0)
                elif kernel == "WhiteKernel":
                    self.gpr_kernel = 1.0 * WhiteKernel(1.0)
                elif kernel == "ExpSineSquared":
                    self.gpr_kernel = ExpSineSquared()
                elif kernel == "RationalQuadratic":
                    self.gpr_kernel = RationalQuadratic(1.0)
                self.kwargs["kernel"] = self.gpr_kernel
                self.kwargs["alpha"] = gpr_alpha
                self.kwargs["normalize_y"] = normalize_y
            self.surrogate = GaussianProcessRegressor(**self.kwargs)
        elif self.name == "decision-tree-regressor":
            try:
                if self.kwargs["max_depth"] == 0:
                    self.kwargs["max_depth"] = None
            except KeyError:
                logger.error("No max_depth")
            self.surrogate = DecisionTreeRegressor(**self.kwargs)
        elif self.name == "SVR":
            self.surrogate = SVR(**self.kwargs)
        # else:
        #     raise Exception('Undefined surr')

    def fit(self, X, y):
        logger.debug(f"X: {X}, y: {y}")
        self.create()
        self.surrogate.fit(X, y)

    def score(self, X, y):
        r_2 = self.surrogate.score(X, y)
        y_pred = self.surrogate.predict(X)
        mse = mean_squared_error(y, y_pred)
        rmse = np.sqrt(mse)
        return r_2, mse, rmse

    def predict(self, X):
        m = self.surrogate.predict(X)
        return m


def get_prediction_uncertanty(model, X, surrogate_name, percentile=90):
    interval_len = []
    if surrogate_name == "random-forest-regressor":
        for x in range(len(X)):
            preds = []
            for pred in model.estimators_:
                prediction = pred.predict(np.array(X[x]).reshape(1, -1))
                preds.append(prediction[0])
            err_down = np.percentile(preds, (100 - percentile) / 2.0)
            err_up = np.percentile(preds, 100 - (100 - percentile) / 2.0)
            interval_len.append(err_up - err_down)
    elif surrogate_name == "GPR":
        y_hat, y_sigma = model.predict(X, return_std=True)
        interval_len = list(y_sigma)
    elif surrogate_name == "decision-tree-regressor":
        raise NotImplementedError
    return interval_len
