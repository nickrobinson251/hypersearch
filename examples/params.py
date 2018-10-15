"""Minimal example of programmatically defining search space."""
from scipy.stats import randint as sp_randint
from skopt.space import Categorical, Integer, Real

from hypersearch import dump, load

params = {
    "max_depth": [3, None],
    "max_features": sp_randint(1, 11),
    "min_samples_split": sp_randint(2, 11),
    "bootstrap": [True, False],
    "criterion": ["gini", "entropy"]
}
dump(params, "params_grid")

params = {
    "max_depth": Categorical([3, None]),
    "max_features": Integer(1, 11),
    "min_samples_split": Integer(2, 11),
    "bootstrap": Integer(0, 1),
    "criterion": Categorical(["gini", "entropy"])
}
dump(params, "params_randomized")

params = {
    "max_depth": [3, None],
    "max_features": [1, 3, 10],
    "min_samples_split": [2, 3, 10],
    "bootstrap": [True, False],
    "criterion": ["gini", "entropy"]
}
dump(params, "params_bayes")
params_bayes = load("params_bayes")
