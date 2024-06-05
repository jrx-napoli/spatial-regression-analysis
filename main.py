import dataset_gen
import experiments

if __name__ == "__main__":
    y, y_name, x, x_names, w, w_name, ds_name = dataset_gen.__dict__["baltimore"]()
    experiments.run(y, y_name, x, x_names, w, w_name, ds_name)
