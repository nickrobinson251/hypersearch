# Hyperparameter Search

A small python library for distributed hyperparameter search.

## Installation

```
git clone this_repo loc
pip install loc
```

## Use case

You have designed and implemented your machine learning models.
You have collected and pre-processed your datasets ready for training. You're ready
to go, except your model has a bunch of hyperparameters: batch size, learning
rate, regularisation term, etc. How can you efficeintly find good settings of
these parameters?

We provide a simple, consistent interface for searching over parameters, whether
training on your local machine, in the cloud, or on a high-perfoormance
computers with a job scheduler.

## Usage

First create and then activate a new conda environment
```
conda env create -f environment.yml
conda activate hyperparam
```

Here are [install instructions for  Anaconda or Miniconda](http://docs.anaconda.com/anaconda/install/), if your machine does not already have it.

### Using a remote machine

If running dask on a remote machine, such as an AWS EC2 instance or a High
Performance Computer, You may still want to setup access to the dask dashboard for
montioring your job.

Ycan set up SSH port-forward to access this dashboard (a HTTP server running
on the remote machine). From your local machine run:
```
ssh -N -L 8787:remotehostname:8787 myusername@remotehost.com
```
where `remotehostname` is the host name of the machine running dask and `myusername@...` is the
publically available address, i.e. the address you used when you first SSH'ed
into the remote machine. `8787` is the default port for the dask dashboard. You
need to make sure this port is not in use on your local machine (e.g. for a
local dask client) or forward to another port.

The (remote) dashboard will then be available to view (locally) at
http://localhost:8787/status

If you are not running on your lcoal machine, e.g. because you are on AWS EC@
instance or a

### Using a HPC cluster

You can set up a configuration file specific to your cluster (see examples/jobqueue.yaml) and connect as shown in examples/cluster.py. Note that the import needs to change depending on your job manager, e.g. `import PBSCluster`.

### Using Kubernetes or Yarn

Please instead see [dask-kubernetes](https://kubernetes.dask.org/en/latest/) or [dask-yarn](https://yarn.dask.org/en/latest/)  rather than using dask-jobqueue. Usage is very similar but these packages will need to be installed.

### Debugging

This is not yet very well tested in general. If you hit a problem (sorry), you
may want to check out [Dask's advice on
debugging](http://docs.dask.org/en/latest/debugging.html).

## Testing

Simply run

```
py.test
```

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