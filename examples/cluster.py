i"""A minimal example of connecting to a PBS cluster and submitting a job."""
from dask_jobqueue import PBSCluster
from dask.distributed import Client
import dask.array as da

cluster = PBSCluster()  # defaults to values set in ~/.config/dask/jobqueue.yml
cluster.scale(10)  # ask for ten workers

client = Client(cluster)  # connect this local process to remote workers
cluster.scale(1)  # ask for one more workers

# run a simple computation  - TODO replace this with hyperparameter search
x = da.random.random((10000, 10000), chunks=(1000, 1000))  # create data
x.persist()  # keep in memory
z = (x - x.mean(axis=0)) / x.std(axis=0)
z.persist()
