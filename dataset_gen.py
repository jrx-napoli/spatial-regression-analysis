import geopandas as gpd
import libpysal
import numpy as np
from libpysal.examples import load_example
from libpysal.weights import Queen


def generate_synthetic_data(data_size):
    np.random.seed(12345)
    x1 = np.random.rand(data_size, 1)
    x2 = np.random.rand(data_size, 1)
    x = np.hstack([x1, x2])
    y = 3 + 2 * x1 + 3 * x2 + np.random.randn(data_size, 1) * 0.5
    return x, y


def baltimore():
    np.set_printoptions(suppress=True)  # prevent scientific format
    ds_name = "baltimore"
    baltimore_example = load_example('Baltimore')
    db = libpysal.io.open(baltimore_example.get_path("baltim.dbf"), 'r')
    df = gpd.read_file(baltimore_example.get_path("baltim.shp"))

    # dependent variable
    y_name = "PRICE"
    y = np.array(db.by_col(y_name)).T
    y.shape = (len(y), 1)

    # exogenous variables
    x_names = ["NROOM", "NBATH", "PATIO", "FIREPL", "AC", "GAR", "AGE", "LOTSZ", "SQFT"]
    x = np.array([db.by_col(var) for var in x_names]).T

    # spatial weights
    w_name = "baltim_q.gal"
    w = Queen.from_dataframe(df)
    w.transform = 'r'

    return y, y_name, x, x_names, w, w_name, ds_name
