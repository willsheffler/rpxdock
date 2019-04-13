from time import perf_counter
import numpy as np
from cppimport import import_hook
from . import bvh_test

from . import bvh


def test_geom():
    # assert bvh_test.test_bvh_test_min()
    # assert bvh_test.test_bvh_test_isect()

    np.random.seed(0)

    xyz1 = np.random.normal(size=(1200, 3)) + [4, 0, 0]
    xyz2 = np.random.normal(size=(1300, 3))
    bvh1 = bvh.make_bvh(xyz1)
    bvh2 = bvh.make_bvh(xyz2)

    tbvh = perf_counter()
    d, i1, i2 = bvh.bvh_min_dist(bvh1, bvh2)
    tbvh = perf_counter() - tbvh
    print("from bvh:", d)
    dtest = np.linalg.norm(xyz1[i1] - xyz2[i2])
    assert np.allclose(d, dtest, atol=1e-5)

    tnp = perf_counter()
    dnp = np.min(np.linalg.norm(xyz1[:, None] - xyz2[None], axis=2))
    tnp = perf_counter() - tnp
    assert np.allclose(dnp, d)
    print("from numpy:", dnp)

    print("bvh acceleration vs naive numpy", tnp / tbvh)
