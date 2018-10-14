from setuptools import setup

setup(name='hypersearch',
      version='0.0.1',
      install_requires=[
          "dask",
          "dask_ml",
          "distributed",
          "scikit-optimize"
        ]
)
