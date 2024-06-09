import argparse


def get_args(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset', type=str, default='synthetic', help='Dataset to be used in experiment')
    parser.add_argument('--n', type=int, default=100, help='Number of observations')
    parser.add_argument('--k', type=int, default=10, help='Number of independent variables')
    parser.add_argument('--regime_k', type=int, default=2, help='Number of independent variables')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed. If defined all random operations will be reproducible')

    return parser.parse_args(argv)
