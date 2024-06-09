import os
import time

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_squared_error
from spreg import OLS, GM_Error, ML_Error, GM_Lag, ML_Lag, GM_Error_Het, GM_Error_Hom, ML_Lag_Regimes, OLS_Regimes, \
    TSLS_Regimes, ML_Error_Regimes, GM_Lag_Regimes, GM_Error_Regimes, GM_Error_Het_Regimes, GM_Error_Hom_Regimes


class SpatialRegressionComparison:
    """
    A class to compare various spatial regression models on a given dataset.

    Attributes:
        X (numpy array): The independent variables.
        y (numpy array): The dependent variable.
        weights (numpy array): Spatial weight's matrix.
        q (numpy array): Instrumental variables for certain models.
        y_end (numpy array): Endogenous variables for regimes models.
        regimes (list): Regimes for each observation.
        y_name (str): Name of the dependent variable.
        x_names (list): Names of the independent variables.
        w_name (str): Name of the weight's matrix.
        regimes_name (str): Name of the regimes variable.
        dataset (DataFrame): The dataset being used.
        dataset_name (str): Name of the dataset.
        models (dict): Dictionary to store the fitted models.
        mse_results (dict): Dictionary to store mean squared error results.
        execution_times (dict): Dictionary to store execution times of models.
        results_path (str): Path to store the results.
    """

    def __init__(self,
                 x,
                 y,
                 w,
                 y_name,
                 x_names,
                 w_name,
                 dataset,
                 dataset_name,
                 q=None,
                 y_end=None,
                 regimes=None,
                 regimes_name=None):
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
        self.mse_results = {}
        self.execution_times = {}
        self.results_path = "results"

    def run_model(self, model_name, model_class, *args):
        """
        Runs a specified spatial regression model and records its execution time.

        Parameters:
            model_name (str): The name of the model to be run.
            model_class (class): The class of the model to be run.
            *args: Additional arguments required by the model.
        """
        start_time = time.time()
        model = model_class(self.y,
                            self.X,
                            *args,
                            name_y=self.y_name,
                            name_x=self.x_names,
                            name_w=self.w_name,
                            name_ds=self.dataset_name)
        end_time = time.time()

        self.models[model_name] = model
        self.execution_times[model_name] = np.round(end_time - start_time, 3)

    def run_all_models(self):
        """
        Runs all spatial regression models.
        """
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
        if self.dataset_name != "SYNTHETIC_DATASET":
            self.run_model('TSLS_Regimes', TSLS_Regimes, self.y_end, self.q, self.regimes, self.weights)
        self.run_model('ML_Lag_Regimes', ML_Lag_Regimes, self.regimes, self.weights)
        self.run_model('ML_Error_Regimes', ML_Error_Regimes, self.regimes, self.weights)
        self.run_model('GM_Lag_Regimes', GM_Lag_Regimes, self.regimes, None, None, self.weights)
        self.run_model('GM_Error_Regimes', GM_Error_Regimes, self.regimes, self.weights)
        self.run_model('GM_Error_Het_Regimes', GM_Error_Het_Regimes, self.regimes, self.weights)
        self.run_model('GM_Error_Hom_Regimes', GM_Error_Hom_Regimes, self.regimes, self.weights)

    def compare_models(self):
        """
        Compares all fitted models based on their mean squared error and execution time, generates reports.
        """
        if not os.path.exists(self.results_path):
            os.makedirs(self.results_path)

        mse_results = {}
        for name, model in self.models.items():
            f_model = open(f"{self.results_path}/{name}.txt", "w")
            f_model.write(model.summary)

            predictions = model.predy
            mse = mean_squared_error(self.y, predictions)
            mse_results[name] = np.round(mse, 3)
        self.mse_results = mse_results

        self._create_reports()
        self._plot_execution_times()
        self._plot_mse_results()

    def _create_reports(self):
        """
        Creates and saves comparison reports for mean squared error and execution time.
        """
        f_comp_error = open(f"{self.results_path}/comparison_error_{self.dataset_name}.txt", "w")
        f_comp_error.write(F"{self.dataset_name}: MEAN SQUARED ERROR COMPARISON\n")
        f_comp_error.write(F"--------------------------------------------------\n\n")

        f_comp_time = open(f"{self.results_path}/comparison_time_{self.dataset_name}.txt", "w")
        f_comp_time.write(F"{self.dataset_name}: EXECUTION TIME COMPARISON [s]\n")
        f_comp_time.write(F"--------------------------------------------------\n\n")

        self.mse_results = dict(sorted(self.mse_results.items(), key=lambda item: item[1]))
        for name, result in self.mse_results.items():
            f_comp_error.write(f"{name}: {np.round(result, 3)}\n")

        self.execution_times = dict(sorted(self.execution_times.items(), key=lambda item: item[1]))
        for name, result in self.execution_times.items():
            f_comp_time.write(f"{name}: {self.execution_times[name]}\n")

    def _plot_execution_times(self):
        """
        Plots a bar chart of execution times.
        """
        plt.figure(figsize=(10, 6))
        plt.bar(self.execution_times.keys(), self.execution_times.values(), color='skyblue')
        plt.xlabel('Model')
        plt.ylabel('Execution Time (seconds)')
        plt.title('Execution Time of Spatial Regression Models')
        plt.xticks(rotation=45)
        plt.tight_layout()

        plt.savefig(f"{self.results_path}/execution_times.png")
        plt.close()

    def _plot_mse_results(self):
        """
        Plots a bar chart of mean squared errors.
        """
        plt.figure(figsize=(10, 6))
        plt.bar(self.mse_results.keys(), self.mse_results.values(), color='skyblue')
        plt.xlabel('Model')
        plt.ylabel('MSE')
        plt.title('Mean Squared Error of Spatial Regression Models')
        plt.xticks(rotation=45)
        plt.tight_layout()

        plt.savefig(f"{self.results_path}/mse_results.png")
        plt.close()
