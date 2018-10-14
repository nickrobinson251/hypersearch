import os
import pytest
from scipy.stats import randint, uniform
from skopt.space import Categorical, Integer, Real
from yaml.parser import ParserError

from hypersearch import parse_params


CURRENT_DIR = os.path.dirname(os.path.realpath(__file__))
GOOD_YAML_FILE = os.path.join(CURRENT_DIR, "params.yaml")
BAD_YAML_FILE = os.path.join(CURRENT_DIR, "bad_format.yaml")


class TestParseParams:
    """Methods for tasking the parse_params functionality."""

    def test_parse_grid(self):
        config1 = parse_params(GOOD_YAML_FILE, "grid")
        config2 = {
            "max_depth": [3, None],
            "max_features": [1,  11],
            "min_samples_split": [2, 11],
            "bootstrap": [0, 1],
            "criterion": ["gini", "entropy"],
            "learning_rate": [1e-06, 10.0]
        }
        assert config1 == config2

    def test_parse_randomized(self):
        config1 = parse_params(GOOD_YAML_FILE, "randomized")
        config2 = {
            "max_depth": [3, None],
            "max_features": randint(1,  11),
            "learning_rate": uniform(1e-06, 10.0)
        }
        assert isinstance(config1["max_depth"], type(config2["max_depth"]))
        assert isinstance(
            config1["learning_rate"], type(config2["learning_rate"]))
        assert isinstance(
            config1["max_features"], type(config2["max_features"]))

    def test_parse_bayes(self):
        config1 = parse_params(GOOD_YAML_FILE, "bayes")
        config2 = {
            "max_depth": Categorical([3, None]),
            "max_features": Integer(1,  11),
            "learning_rate": Real(1e-06, 10.0)
        }
        assert type(config1["max_depth"]) == type(config2["max_depth"])
        assert isinstance(
            config1["learning_rate"], type(config2["learning_rate"]))
        assert isinstance(
            config1["max_features"], type(config2["max_features"]))

    def test_parse_with_unkown_mathod(self):
        with pytest.raises(ValueError):
            parse_params(GOOD_YAML_FILE, "foobar")

    def test_badly_formatted_yaml(self):
        with pytest.raises(ParserError):
            parse_params(BAD_YAML_FILE, "grid")
