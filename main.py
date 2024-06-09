import datasets
from experiments import SpatialRegressionComparison

if __name__ == "__main__":
    y, y_name, x, x_names, w, w_name, regimes, regimes_name, dataset, ds_name = datasets.__dict__["baltimore"]()

    src = SpatialRegressionComparison(x=x, y=y, w=w, regimes=regimes, y_name=y_name, x_names=x_names, w_name=w_name,
                                      regimes_name=regimes_name,
                                      dataset=dataset,
                                      dataset_name=ds_name)
    src.run_all_models()

    mse_results, execution_times = src.compare_models()
    print("Mean Squared Errors:", mse_results)
    print("Execution Times [s]:", execution_times)
