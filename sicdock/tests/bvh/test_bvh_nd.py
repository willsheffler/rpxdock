from time import perf_counter
import numpy as np
from cppimport import import_hook
import homog as hm
from sicdock.bvh import bvh_nd
from scipy.spatial.distance import cdist


def test_bvh_nd():
    nhit, ntot = 0, 0
    for iter in range(10):
        pts = np.random.randn(1_000, 7) + [2, 2, 2, 2, 2, 2, 2]
        bvh = bvh_nd.create_bvh7(pts)
        assert np.allclose(bvh.com(), np.mean(pts, axis=0))

        pts2 = np.random.randn(1_000, 7)
        bvh2 = bvh_nd.create_bvh7(pts2)

        for thresh in np.arange(0.1, 2, 0.1):
            isect1 = bvh_nd.bvh_isect7(bvh, bvh2, thresh)
            isect2 = bvh_nd.naive_isect7(bvh, bvh2, thresh)
            assert isect1 == isect2
            nhit += isect1
            ntot += 1
    print("frac", nhit / ntot)


def DISABLED_test_bvh_ori():
    ntot, nhit, ttot = 0, 0, 0
    thresh = 0.03
    for i in range(3):
        oris1 = hm.rand_xform(1000)[:, :3, :3]
        bvh1 = bvh_nd.create_bvh9(oris1.reshape(-1, 9))
        oris2 = hm.rand_xform(1000)[:, :3, :3]
        bvh2 = bvh_nd.create_bvh9(oris2.reshape(-1, 9))
        for j in range(10):
            r1, r2 = hm.rand_xform(2)[:, :3, :3]
            t = perf_counter()
            isecta = bvh_nd.bvh_isect_ori(bvh1, bvh2, r1, r2, thresh)
            ttot += perf_counter() - t
            isectb = bvh_nd.naive_isect_ori(bvh1, bvh2, r1, r2, thresh)
            assert isecta == isectb
            nhit += isecta
            ntot += 1
    print(f"frac {nhit / ntot}, {int(ntot / ttot):,}/s")


def DISABLED_test_bvh_xform():
    ntot, nhit, ttot = 0, 0, 0
    thresh = 0.5
    for i in range(3):
        xforms1 = hm.rand_xform(1000, cart_sd=0.5)
        xforms1[:, :3, 3] += 1.0
        xforms1 = xforms1.reshape(-1, 16)[:, :12]
        xforms2 = hm.rand_xform(1000, cart_sd=0.5).reshape(-1, 16)[:, :12]
        bvh1 = bvh_nd.create_bvh12(xforms1)
        bvh2 = bvh_nd.create_bvh12(xforms2)
        for j in range(10):
            x1, x2 = hm.rand_xform(2)
            t = perf_counter()
            isecta = bvh_nd.bvh_isect_xform(bvh1, bvh2, x1, x2, thresh)
            ttot += perf_counter() - t
            isectb = bvh_nd.naive_isect_xform(bvh1, bvh2, x1, x2, thresh)
            assert isecta == isectb
            nhit += isecta
            ntot += 1
    print(f"frac {nhit / ntot}, {int(ntot / ttot):,}/s")


def test_bvh_quat_min():
    N = 100
    o = hm.rand_xform(1000)[:, :3, :3].copy()
    assert np.allclose(np.linalg.det(o), 1)
    bvh_ohier = bvh_nd.create_bvh_quat(o.reshape(-1, 9))
    samp = hm.rand_xform(N)[:, :3, :3]
    for j in range(N):
        d1 = bvh_nd.bvh_min_one_quat(bvh_ohier, samp[j])
        d2 = bvh_nd.naive_min_one_quat(bvh_ohier, samp[j])[0]
        if d1 != d2:
            print(d1, d2)
        assert np.allclose(d1, d2)


if __name__ == "__main__":
    # test_bvh_nd()
    # test_bvh_ori()
    # test_bvh_xform()
    test_bvh_quat_min()
