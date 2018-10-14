"""Search for Hyperparameters."""
import argparse
import dask_ml.joblib
import logging
from dask.distributed import Client, progress
from scipy.stats import randint as sp_randint
from sklearn.datasets import load_digits
from sklearn.ensemble import RandomForestClassifier
from sklearn.externals import joblib
from skopt.space import Real, Categorical, Integer

from hypersearch import search

def parse_args():
    parser = argparse.ArgumentParser(
        description="Optimize model hyperparameters.")
    parser.add_argument("filepath", type=str,
                        help="Complete path to file where model will be saved.")
    parser.add_argument("--method", choices=['bayes', 'grid', 'randomized'],
                        default="randomized")
    return parser.parse_args()

if __name__ == '__main__':
    LOGGER = logging.getLogger(__name__)
    LOGGER.addHandler(logging.StreamHandler)

    CLIENT = Client()

    digits = load_digits()
    X, y = digits.data, digits.target
    model = RandomForestClassifier(n_estimators=20)

    args = parse_args()
    if args.method == "randomized":
        params = {
            "max_depth": [3, None],
            "max_features": sp_randint(1, 11),
            "min_samples_split": sp_randint(2, 11),
            "bootstrap": [True, False],
            "criterion": ["gini", "entropy"]
        }
    elif args.method == "bayes":
        params = {
            "max_depth": Categorical([3, None]),
            "max_features": Integer(1, 11),
            "min_samples_split": Integer(2, 11),
            "bootstrap": Integer(0, 1),
            "criterion": Categorical(["gini", "entropy"])
        }
    else:
        params = {
            "max_depth": [3, None],
            "max_features": [1, 3, 10],
            "min_samples_split": [2, 3, 10],
            "bootstrap": [True, False],
            "criterion": ["gini", "entropy"]
        }

    with joblib.parallel_backend('dask.distributed'):
        result = search(model, X, y, params, method=args.method)
        joblib.dump(result, args.filepath)
