import _pickle, os, multiprocessing


def load(f):
    with open(f, "rb") as inp:
        return _pickle.load(inp)


def dump(obj, f):
    os.makedirs(os.path.dirname(f), exist_ok=True)
    with open(f, "wb") as out:
        return _pickle.dump(obj, out)


def cpu_count():
    try:
        return int(os.environ["SLURM_CPUS_ON_NODE"])
    except:
        return multiprocessing.cpu_count()
