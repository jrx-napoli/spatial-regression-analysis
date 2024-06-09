import geopandas as gpd
import libpysal
import numpy as np
import pandas as pd
from libpysal.examples import load_example
from libpysal.weights import Queen, lat2W


def generate_synthetic_data(n=100, k=3, regime_k=2, seed=None):
    """
    Generates synthetic data for spatial regression models.

    Parameters:
    - n: int, number of observations
    - k: int, number of independent variables
    - regime_k: int, number of regimes
    - seed: int, random seed for reproducibility

    Returns:
    - X: numpy array, independent variables
    - y: numpy array, dependent variable
    - weights: pysal.weights.W, spatial weights
    - regimes: list, regimes for each observation
    """
    if seed is not None:
        np.random.seed(seed)

    # Generate synthetic coordinates for a grid
    x_coords = np.random.uniform(low=0, high=10, size=n)
    y_coords = np.random.uniform(low=0, high=10, size=n)

    # Create a DataFrame to hold coordinates
    coords = pd.DataFrame({'x': x_coords, 'y': y_coords})

    # Generate a lattice spatial weights matrix
    w_name = "SYNTHETIC_W"
    weights = lat2W(int(np.sqrt(n)), int(np.sqrt(n)))

    # Generate synthetic independent variables
    x_names = [f"X{i}" for i in range(k)]
    X = np.random.randn(n, k)

    # Generate a synthetic dependent variable
    y_name = "Y_VAR"
    beta = np.random.randn(k)
    y = X @ beta + np.random.randn(n)

    # Generate regimes
    regimes_name = "SYNTHETIC_REGIMES"
    regimes = np.random.choice(range(1, regime_k + 1), size=n)

    db = None
    ds_name = "SYNTHETIC_DATASET"

    return y, y_name, X, x_names, weights, w_name, regimes, regimes_name, db, ds_name


def baltimore():
    np.set_printoptions(suppress=True)  # prevent scientific format
    ds_name = "Baltimore"
    example = load_example('Baltimore')
    db = libpysal.io.open(example.get_path("baltim.dbf"), 'r')
    df = gpd.read_file(example.get_path("baltim.shp"))

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

    regimes_name = "CITCOU"
    regimes = db.by_col(regimes_name)

    return y, y_name, x, x_names, w, w_name, regimes, regimes_name, db, ds_name


def columbus():
    np.set_printoptions(suppress=True)  # prevent scientific format
    ds_name = 'Columbus'
    example = load_example('Columbus')
    db = libpysal.io.open(example.get_path("columbus.dbf"), 'r')
    df = gpd.read_file(example.get_path("columbus.shp"))

    y_name = 'HOVAL'
    y = np.array(db.by_col(y_name)).T
    y.shape = (len(y), 1)

    x_names = ['INC', 'CRIME']
    x = np.array([db.by_col(var) for var in x_names]).T

    w_name = "columbus.gal"
    w = Queen.from_dataframe(df)
    w.transform = 'r'

    return y, y_name, x, x_names, w, w_name, db, ds_name
