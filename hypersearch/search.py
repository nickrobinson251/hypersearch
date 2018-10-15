"""Search for Hyperparameters."""
from dask.distributed import Client, progress
from dask_ml.model_selection import GridSearchCV, RandomizedSearchCV
from pandas import DataFrame as df
from scipy.stats import randint as sp_randint
from sklearn.datasets import load_digits
from sklearn.ensemble import RandomForestClassifier
from sklearn.externals.joblib import parallel_backend
from skopt import BayesSearchCV
from skopt.space import Real, Categorical, Integer
from time import time
import dask_ml.joblib
import logging


LOGGER = logging.getLogger()


def search(model,
           X,
           y,
           params,
           method="randomized",
           n_iter=30,
           cv=5,
           **kwargs):
    """Run a cross-validated search for hyperparameters."""
    if method.lower() == "randomized":
        search = RandomizedSearchCV(
            model, param_distributions=params, n_iter=n_iter, cv=cv)
    elif method.lower() == "grid":
        search = GridSearchCV(
            model, param_grid=params, cv=cv)
    elif method.lower() == "bayes":
        search = BayesSearchCV(
            model, search_spaces=params, n_iter=n_iter, cv=cv)
    else:
        message = ("'method' must be either 'randomized', 'grid' or 'bayes'."
                   " Got method='{}'".format(method))
        LOGGER.error(message)
        raise ValueError(message)

    method_name = method.capitalize() + "SearchCV"
    LOGGER.info("Beginning " + method_name)
    when_started = time()

    progress(search.fit(X, y))

    total_time = time() - when_started
    n_settings = len(search.cv_results_['params'])
    LOGGER.warn("{} took {:.2f} seconds for {} candidates parameter settings."
                .format(method_name, total_time, n_settings))
    return search
