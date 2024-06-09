import os
import time

import numpy as np
from sklearn.metrics import mean_squared_error
from spreg import OLS, GM_Error, ML_Error, GM_Lag, ML_Lag, GM_Error_Het, GM_Error_Hom, ML_Lag_Regimes, OLS_Regimes, \
    TSLS_Regimes, ML_Error_Regimes, GM_Lag_Regimes, GM_Error_Regimes, GM_Error_Het_Regimes, GM_Error_Hom_Regimes, \
    sur_dictxy


class SpatialRegressionComparison:
    def __init__(self, x, y, w, y_name, x_names, w_name, dataset, dataset_name, q=None, y_end=None, regimes=None, regimes_name=None):
        self.X = x
        self.y = y
        self.weights = w
        self.q = q if q is not None else np.ones((self.X.shape[0], 1))
        self.y_end = y_end if y_end is not None else np.ones((self.X.shape[0], 1))
        self.regimes = regimes if regimes is not None else []
        self.y_name = y_name
        self.x_names = x_names
        self.w_name = w_name
        self.regimes_name = regimes_name
        self.dataset = dataset
        self.dataset_name = dataset_name

        self.models = {}
        self.execution_times = {}
        self.results_path = "results"

    def run_model(self, model_name, model_class, *args):
        start_time = time.time()
        model = model_class(self.y, self.X, *args, name_y=self.y_name, name_x=self.x_names, name_w=self.w_name, name_ds=self.dataset_name)
        end_time = time.time()

        self.models[model_name] = model
        self.execution_times[model_name] = np.round(end_time - start_time, 3)

    def run_all_models(self):
        # Spatial regression models
        self.run_model('OLS', OLS)
        self.run_model('ML_Lag', ML_Lag, self.weights)
        self.run_model('ML_Error', ML_Error, self.weights)
        self.run_model('GM_Lag', GM_Lag, None, None, self.weights)
        self.run_model('GM_Error', GM_Error, self.weights)
        self.run_model('GM_Error_Het', GM_Error_Het, self.weights)
        self.run_model('GM_Error_Hom', GM_Error_Hom, self.weights)

        # Regimes models
        self.run_model('OSL_Regimes', OLS_Regimes, self.regimes)
        self.run_model('TSLS_Regimes', TSLS_Regimes, self.y_end, self.q, self.regimes, self.weights)
        self.run_model('ML_Lag_Regimes', ML_Lag_Regimes, self.regimes, self.weights)
        self.run_model('ML_Error_Regimes', ML_Error_Regimes, self.regimes, self.weights)
        self.run_model('GM_Lag_Regimes', GM_Lag_Regimes, self.regimes, None, None, self.weights)
        self.run_model('GM_Error_Regimes', GM_Error_Regimes, self.regimes, self.weights)
        self.run_model('GM_Error_Het_Regimes', GM_Error_Het_Regimes, self.regimes, self.weights)
        self.run_model('GM_Error_Hom_Regimes', GM_Error_Hom_Regimes, self.regimes, self.weights)

        # Seemingly-Unrelated Regressions
        bigy, bigX, bigyvars, bigXvars = self._setup_sur()

    def _setup_sur(self):
        bigy, bigX, bigyvars, bigXvars = sur_dictxy(self.dataset, [self.y_name], self.x_names)
        return bigy, bigX, bigyvars, bigXvars

    def compare_models(self):
        mse_results = {}
        for name, model in self.models.items():

            if not os.path.exists(self.results_path):
                os.makedirs(self.results_path)
            f = open(f"{self.results_path}/{name}.txt", "w")
            f.write(model.summary)

            predictions = model.predy
            mse = mean_squared_error(self.y, predictions)
            mse_results[name] = np.round(mse, 3)

        return mse_results, self.execution_times
