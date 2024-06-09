import sys

import datasets
from options import get_args
from experiments import SpatialRegressionComparison

if __name__ == "__main__":
    args = get_args(sys.argv[1:])

    if args.dataset == "synthetic":
        y, y_name, x, x_names, w, w_name, regimes, regimes_name, dataset, ds_name \
            = datasets.generate_synthetic_data(n=args.n, k=args.k, regime_k=args.regime_k, seed=args.seed)
    else:
        y, y_name, x, x_names, w, w_name, regimes, regimes_name, dataset, ds_name = datasets.__dict__[args.dataset]()

    src = SpatialRegressionComparison(x=x, y=y, w=w, regimes=regimes, y_name=y_name, x_names=x_names, w_name=w_name,
                                      regimes_name=regimes_name,
                                      dataset=dataset,
                                      dataset_name=ds_name)
    src.run_all_models()
    src.compare_models()
