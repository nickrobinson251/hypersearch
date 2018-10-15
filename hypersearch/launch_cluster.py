from dask.distributed import LocalCluster


def launch_cluster(cluster_type="local", scale=2):
    """Launch Dask on a cluster.

    Parameters
    ----------
    cluster_type : str
    scale : int
        Request this number of workers
    """
    if cluster_type.lower() == "local":
        return LocalCluster()

    import dask_jobqueue as jobq
    name2class = {
        "lsf": jobq.LSFCluster,
        "moab": jobq.MoabCluster,
        "pbs": jobq.PBSCluster,
        "sge": jobq.SGECluster,
        "slurm": jobq.SLURMCluster,
    }
    name = cluster_type.lower()
    try:
        cluster = name2class[name]()
    except ValueError as err:
        message = (err + "\n" + "Try setting a configuration for the cluster"
                   " by editing '~/.config/dask/jobqueue.yaml'.")
        raise ValueError(message)
    cluster.scale(scale)
    return cluster
