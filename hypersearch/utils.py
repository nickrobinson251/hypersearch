"""Wrap  saving and loading, to make useage easier and in case we swap joblib."""
from sklearn.externals import joblib

def dump(obj, filename):
    return joblib.dump(obj, filename)

def load(obj, filename):
    return joblib.load(filename)
