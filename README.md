# Hyperparameter Search

A small python library for distributed hyperparameter search.

NOTE: THIS IS WORK IN PROGRESS AND WRITTEN IN ONE DAY. DO NOT USE IN PRODUCTION!

## Installation

```
git clone https://github.com/nickrobinson251/hypersearch.git
pip install hypersearch
```

## Use case

You have designed and implemented your machine learning models.
You have collected and pre-processed your datasets ready for training. You're ready
to go, except your model has a bunch of hyperparameters: batch size, learning
rate, regularisation term, etc. How can you efficeintly find good settings of
these parameters?

We aim to provide a simple, consistent interface for searching over parameters, whether
training on your local machine, in the cloud, or on a High
Performance Computer (HPC) cluster with a job scheduler.

## Usage

First create and then activate a new conda environment
```
conda env create -f environment.yml
conda activate hypersearch
```

Here are [install instructions for  Anaconda or Miniconda](http://docs.anaconda.com/anaconda/install/), if your machine does not already have it.

Then assuming you have a `model` and labelled dataset (`X`, `y`) ready to train, you can use
the `search` function:

```python
from hypersearch import search, dump

method = "Randomized"  # could also be "Grid" or "Bayes"  
result = search(model, X, y, params, method=method)
dump(result, "results_object")  # serialise to disk

print(results.best_params_, results.best_score_)
model_optimized = results.best_estimator_
```
`results` is a exactly the same object as if you were using scikit-learn.
For general advice on hyperparameter tuning and how to use the results,
see the [sklearn user guide](http://scikit-learn.org/stable/modules/grid_search.html#grid-search).

### Specifying hyperparameter search space

For this to work, we need to specify a search space (`params` above). This can 
be done by either:
1. Using a config file - this provides a consistent interface, allowing you to write one config 
and use it with any search method. The config must be of the format given in `examples/params.yaml`.
2. Progammitcally specify our own distribution over parameters - this allows for arbitrary distributions over
integer and real-valued hyperparameters, useful for "Randomized" and "bayes" search. An example is given in 
`examples/params.py`.

### Choosing a search method

Three search methods are currently available. We essentially provide a wrapper that ensures
distributed training, please see the documentation of the underlying methods:
1. "Grid" : [GridSearchCV](http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html#sklearn.model_selection.GridSearchCV)
2. "Randomized" : [RandomizedSearchCV](http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.RandomizedSearchCV.html)
3. "Bayes" : [BayesSearchCV](https://scikit-optimize.github.io/#skopt.BayesSearchCV)

### Using a remote machine

Hypersearch is powered by dask, which provides a useful web dashboard for
montioring your jobs. If running dask on a remote machine, such as an AWS EC2
instance or a HPC, you can set up SSH port-forward to access this dashboard
(a HTTP server running on the remote machine). From your local machine run:
```
ssh -N -L 8787:remotehostname:8787 myusername@remotehost.com
```
where `remotehostname` is the host name of the machine running hypersearch/dask
and `myusername@...` is the publically available address,
i.e. the address you used when you first SSH'ed into the remote machine.
`8787` is the default port for the dask dashboard. You
need to make sure this port is not in use on your local machine (e.g. for a
local dask client) or forward to another port.

The (remote) dashboard will then be available to view (locally) at
http://localhost:8787/status

If you are not running on your lcoal machine, e.g. because you are on AWS EC@
instance or a

### Using a HPC cluster

You can set up a configuration file specific to your cluster (see `examples/jobqueue.yaml`)
and connect as shown in `examples/cluster.py` or `examples/optimize_hyperparameters.py`.

### Using Kubernetes or Yarn

Please instead see [dask-kubernetes](https://kubernetes.dask.org/en/latest/) or [dask-yarn](https://yarn.dask.org/en/latest/) rather than using dask-jobqueue. Usage is very similar but these packages will need to be installed.

### Debugging

This is not yet very well tested in general. Sorry! If you hit a problem, you
may find [Dask's advice ondebugging](http://docs.dask.org/en/latest/debugging.html) helpful.

## Testing

Simply run

```
pytest
```
Unfortunately, we currently only test the parsing of the parameter config file! So, be careful, and monitor your comuting resources manually (for now). 

### Similar packages

This package uses [dask](https://docs.dask.org/en/latest/) and [scikit-optimize](https://scikit-optimize.github.io/), because they provide scalable "drop-in"
replacement for scikit-learn hyperparameter search methods. 
This  means we canuse scikit-learn models (and any models that follow the scikit-learn API by
implemented `fit` and `score` methods). 
Dask was developed partly for [our use-case](http://docs.dask.org/en/latest/use-cases.html#scikit-learn-or-joblib-user).

Here are alternative open-source packages providing (distributed) hyperparameter search and why I didn't use them:
- [GPyOpt](https://github.com/SheffieldML/GPyOpt)
    - Documentation on parallel (multi-core) and distributed (cluster)
      optimization is not great, see: [parallel](https://nbviewer.jupyter.org/github/SheffieldML/GPyOpt/blob/devel/manual/GPyOpt_parallel_optimization.ipynb),
      [distributed](https://nbviewer.jupyter.org/github/SheffieldML/GPyOpt/blob/master/manual/GPyOpt_external_objective_evaluation.ipynb),
      and
      [this discussion](https://github.com/SheffieldML/GPyOpt/issues/172)
- [GPflowOpt](https://github.com/GPflow/GPflowOpt)
    - Again, distributed functionality is not well documented
    - Based on Tensorflow… which adds complexity
- [HyperOpt](https://github.com/hyperopt/hyperopt)
    - Requires mongodb and did not want  to involve a database
- [HyperOpt-sklearn](https://github.com/hyperopt/hyperopt-sklearn)
     - Reimplements sklearn models whereas we want to be able to define our own models and think about hyperparamer optimization afterwards.
- [Ray-tune](https://github.com/ray-project/ray/tree/master/python/ray/tune)
    - An alternative to Dask that I am less familiar with
    - The Ray-tune package is  new and still in development
    - It does however have the hyperband algorithm which seems [a promising
      choice](http://www.argmin.net/2016/06/23/hyperband/) but I prioritised
      having easy distributed use over having an optimal search algorithm.
