"""Wrappers for saving and loading, just to make useage a bit friendlier."""
from sklearn.externals import joblib

def dump(obj, filename):
    return joblib.dump(obj, filename)

def load(obj, filename):
    return joblib.load(obj, filename)
