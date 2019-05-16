from sicdock.util import cpu_count, Bunch
import numpy as np


def defaults():
    return Bunch(
        nworker=cpu_count(),
        max_trim=100,
        nout=10,
        nresl=5,
        wts=Bunch(plug=1.0, hole=1.0, ncontact=0.01, rpx=1.0),
        clashdis=3.5,
        beam_size=1e5,
        rmscut=3.0,
        max_longaxis_dot_z=1.000001,
        multi_iface_summary=np.min,
    )
