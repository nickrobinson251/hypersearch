"""Search for Hyperparameters."""
import argparse
import logging
import os
import dask_ml.joblib  # think this is needed to regiaster dask joblib context
from dask.distributed import Client, progress
from sklearn.datasets import load_digits
from sklearn.ensemble import RandomForestClassifier
from sklearn.externals import joblib

from hypersearch import search, dump, launch_cluster, parse_params


CURRENT_DIR = os.path.dirname(os.path.realpath(__file__))


def parse_args():
    parser = argparse.ArgumentParser(
        description="Optimize model hyperparameters.")
    parser.add_argument("filepath", type=str,
                        help="Path to file where model will be saved.")
    parser.add_argument("--method", choices=['bayes', 'grid', 'randomized'],
                        default="randomized",
                        help="Hyperparameter search method.")
    parser.add_argument("--params", type=str,
                        help="Path to yaml file defining search space.")
    parser.add_argument("--cluster",
                        choices=['local', 'LSF', 'Moab', 'PBS', 'SGE', 'Slurm'],
                        default="local",
                        help="Type of cluster to launch.")
    parser.add_argument("--scale", type=int,
                        help="Number of cluster workers to requesat.")
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    # replace with your data
    digits = load_digits()
    X, y = digits.data, digits.target

    # replace with your model
    model = RandomForestClassifier(n_estimators=20)

    # replace with your hyperparameter search config
    config = os.path.join(CURRENT_DIR, "params.yaml")
    params = parse_params(config, args.method)

    # set up context manager for distributing computations
    LOGGER = logging.getLogger(__name__)
    CLUSTER = launch_cluster(args.cluster, args.scale)
    CLIENT = Client(CLUSTER)
    LOGGER.warn("Web dashboard now running at http://localhost:8787/status")
    with joblib.parallel_backend('dask.distributed'):
        result = search(model, X, y, params, method=args.method)
        # try to get progress bar in terminal; prefer web daskboad
        progress(result)
        dump(result, args.filepath)
