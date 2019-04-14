from time import perf_counter
import numpy as np
from cppimport import import_hook

from tcdock.bvh import bvh_test
from tcdock.bvh import bvh
import homog as hm


def test_bvh_isect():
    # assert bvh_test.test_bvh_test_isect()

    print()
    thresh = 0.01

    totbvh, totnaive = 0, 0

    for i in range(10):
        xyz1 = np.random.rand(1000, 3).astype("f4") + [0.9, 0.9, 0]
        xyz2 = np.random.rand(1000, 3).astype("f4")
        tcre = perf_counter()
        bvh1 = bvh.make_bvh(xyz1)
        bvh2 = bvh.make_bvh(xyz2)
        tcre = perf_counter() - tcre

        pos1 = hm.htrans([0.9, 0.9, 0.9])
        pos2 = np.eye(4)

        tbvh = perf_counter()
        clash1 = bvh.isect_bvh(bvh1, bvh2, thresh)
        tbvh = perf_counter() - tbvh

        tn = perf_counter()
        clash2 = bvh.isect_naive(bvh1, bvh2, thresh)
        tn = perf_counter() - tn

        assert clash1 == clash2

        print(f"{i:3} clash {clash1:1} {tn / tbvh:8.2f}, {tn:1.6f}, {tbvh:1.6f}")

        totbvh += tbvh
        totnaive += tn

    print("total times", totbvh, totnaive / totbvh, totnaive)


def test_bvh_isect_xform():
    # assert bvh_test.test_bvh_test_isect()

    print()
    thresh = 0.02

    totbvh, totnaive = 0, 0

    xyz1 = np.random.rand(1000, 3).astype("f4") - [0.5, 0.5, 0.5]
    xyz2 = np.random.rand(1000, 3).astype("f4") - [0.5, 0.5, 0.5]
    tcre = perf_counter()
    bvh1 = bvh.make_bvh(xyz1)
    bvh2 = bvh.make_bvh(xyz2)
    tcre = perf_counter() - tcre

    for i in range(10):

        pos1 = hm.rand_xform(cart_sd=0.7)
        pos2 = np.eye(4)

        tbvh = perf_counter()
        clash1 = bvh.isect_bvh_pos(bvh1, bvh2, thresh, pos1, pos2)
        tbvh = perf_counter() - tbvh

        tn = perf_counter()
        clash2 = bvh.isect_naive_pos(bvh1, bvh2, thresh, pos1, pos2)
        tn = perf_counter() - tn

        assert clash1 == clash2

        print(f"{i:3} clash {clash1:1} {tn / tbvh:8.2f}, {tn:1.6f}, {tbvh:1.6f}")

        totbvh += tbvh
        totnaive += tn

    print("total times", totbvh, totnaive / totbvh, totnaive)


def test_bvh_min():
    assert bvh_test.test_bvh_test_min()

    xyz1 = np.random.rand(5000, 3).astype("f4") + [0.9, 0.9, 0.0]
    xyz2 = np.random.rand(5000, 3).astype("f4")
    tcre = perf_counter()
    bvh1 = bvh.make_bvh(xyz1)
    bvh2 = bvh.make_bvh(xyz2)
    tcre = perf_counter() - tcre

    tbvh = perf_counter()
    d, i1, i2 = bvh.min_dist_bvh(bvh1, bvh2)
    tbvh = perf_counter() - tbvh
    dtest = np.linalg.norm(xyz1[i1] - xyz2[i2])
    assert np.allclose(d, dtest, atol=1e-6)

    # tnp = perf_counter()
    # dnp = np.min(np.linalg.norm(xyz1[:, None] - xyz2[None], axis=2))
    # tnp = perf_counter() - tnp

    tn = perf_counter()
    dn = bvh.min_dist_naive(bvh1, bvh2)
    tn = perf_counter() - tn

    print()
    print("from bvh:  ", d)
    print("from naive:", dn)
    assert np.allclose(dn, d, atol=1e-6)

    print(f"tnaivecpp {tn:5f} tbvh {tbvh:5f} tbvhcreate {tcre:5f}")
    print("bvh acceleration vs naive", tn / tbvh)
    assert tn / tbvh > 100


def test_bvh_min_xform():

    xyz1 = np.random.rand(1000, 3).astype("f4") - [0.5, 0.5, 0.5]
    xyz2 = np.random.rand(1000, 3).astype("f4") - [0.5, 0.5, 0.5]
    tcre = perf_counter()
    bvh1 = bvh.make_bvh(xyz1)
    bvh2 = bvh.make_bvh(xyz2)
    tcre = perf_counter() - tcre
    print()
    totbvh, totnaive = 0, 0
    for i in range(10):
        pos1 = hm.rand_xform(cart_sd=0.7)
        pos2 = np.eye(4)

        tbvh = perf_counter()
        d, i1, i2 = bvh.min_dist_bvh_pos(bvh1, bvh2, pos1, pos2)
        tbvh = perf_counter() - tbvh
        dtest = np.linalg.norm(pos1 @ hm.hpoint(xyz1[i1]) - pos2 @ hm.hpoint(xyz2[i2]))
        assert np.allclose(d, dtest, atol=1e-6)

        tn = perf_counter()
        dn = bvh.min_dist_naive_pos(bvh1, bvh2, pos1, pos2)
        tn = perf_counter() - tn
        assert np.allclose(dn, d, atol=1e-6)

        print(
            f"tnaivecpp {tn:1.6f} tbvh {tbvh:1.6f} tcpp/tbvh {tn/tbvh:8.1f}",
            np.linalg.norm(pos1[:3, 3]),
            dtest - d,
        )
        totnaive += tn
        totbvh += tbvh

    print("total times", totbvh, totnaive / totbvh, totnaive, f"tcre {tcre:2.4f}")


if __name__ == "__main__":
    test_bvh_isect_xform()
