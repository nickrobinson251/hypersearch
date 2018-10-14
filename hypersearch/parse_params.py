"""Utility functions for loading hyperparameter search space from yaml file

This aims to allow you to write a single yaml file and easily switching between
search methods.
"""
import numpy as np
import yaml
from scipy.stats import randint, uniform
from skopt.space import Categorical, Integer, Real


def parse_params(filepath, search_method):
    """Create dict specifying hyperparameter search space from yaml file.

    Parameters
    ----------
    filepath : str
        Full path to yaml configuration file
    search_method : str
        One of "grid", "bayes" or "randomized"

    Example
    -------
    ```
    # example.yaml
    Categorical:
        hyperparam1:
          a
          b
    Integer:
        hyperparam1:  # range
          -5
          5
    Real:
        hyperparam3
          1.0
          5.0
    ```
    """
    if search_method.lower() not in ("bayes", "grid", "randomized"):
        message = ("'search_method' must be 'randomized', 'grid' or 'bayes'."
                   " Got method='{}'".format(search_method))
        raise ValueError(message)

    with open(str(filepath), "r") as f:
        config = yaml.safe_load(f)

    if search_method == "grid":
        return _to_lists(config)

    elif search_method == "randomized":
        types = {"categorical": list,
                 "integer": randint,
                 "real": uniform}

    elif search_method == "bayes":
        types = {"categorical": Categorical,
                 "integer": Integer,
                 "real": Real}

    return _to_distributions(config, types)


def _to_lists(config):
    """Convert config loaded from yaml into dict of lists."""
    dimensions = {}
    for param_type, params in config.items():
        param_type = param_type.lower()
        for param in params:
            if param_type == "categorical":
                dimension = {name.lower(): [_handle_nones(v) for v in values]
                             for name, values in param.items()}
            elif param_type == "integer":
                dimension = {name.lower(): [int(v) for v in values]
                             for name, values in param.items()}
            elif param_type == "real":
                dimension = {name.lower(): [float(v) for v in values]
                             for name, values in param.items()}
            dimensions.update(dimension)
    return dimensions


def _to_distributions(config, types):
    """Convert config loaded from yaml into dict of distributions.

    Parameters
    ----------
    config : dict
    types : dict
        Mapping from "type" to a function which returns a distribution
    """
    dimensions = {}
    for param_type, params in config.items():
        param_type = param_type.lower()
        for param in params:
            domain = types[param_type]
            if param_type == "categorical":
                for name, values in param.items():
                    values = [_handle_nones(v) for v in values]
                    dimensions.update({name.lower(): domain(values)})
            elif param_type == "integer":
                for name, values in param.items():
                    values = np.array(values).astype(int)
                    dimension = {name.lower(): domain(min(values), max(values))}
                    dimensions.update(dimension)
            elif param_type == "real":
                for name, values in param.items():
                    values = np.array(values).astype(float)
                    dimension = {name.lower(): domain(min(values), max(values))}
                    dimensions.update(dimension)
    return dimensions

def _handle_nones(item):
    return item if item != "None" else None
