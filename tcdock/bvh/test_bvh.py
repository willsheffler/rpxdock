from time import perf_counter
import numpy as np
from cppimport import import_hook

# import cppimport

# cppimport.set_quiet(False)

from tcdock.bvh import bvh_test
from tcdock.bvh import bvh
import homog as hm


def test_bvh_isect_cpp():
    assert bvh_test.test_bvh_test_isect()


def test_bvh_isect_fixed():
    print()
    mindist = 0.01

    totbvh, totnaive = 0, 0

    for i in range(10):
        xyz1 = np.random.rand(1000, 3) + [0.9, 0.9, 0]
        xyz2 = np.random.rand(1000, 3)
        tcre = perf_counter()
        bvh1 = bvh.bvh_create(xyz1)
        bvh2 = bvh.bvh_create(xyz2)
        tcre = perf_counter() - tcre

        pos1 = hm.htrans([0.9, 0.9, 0.9])
        pos2 = np.eye(4)

        tbvh = perf_counter()
        clash1 = bvh.bvh_isect_fixed(bvh1, bvh2, mindist)
        tbvh = perf_counter() - tbvh

        tn = perf_counter()
        clash2 = bvh.naive_isect_fixed(bvh1, bvh2, mindist)
        tn = perf_counter() - tn

        assert clash1 == clash2

        print(f"{i:3} clash {clash1:1} {tn / tbvh:8.2f}, {tn:1.6f}, {tbvh:1.6f}")

        totbvh += tbvh
        totnaive += tn

    print("total times", totbvh, totnaive / totbvh, totnaive)


def test_bvh_isect():
    print()
    mindist = 0.02

    totbvh, totnaive = 0, 0

    xyz1 = np.random.rand(1000, 3) - [0.5, 0.5, 0.5]
    xyz2 = np.random.rand(1000, 3) - [0.5, 0.5, 0.5]
    tcre = perf_counter()
    bvh1 = bvh.bvh_create(xyz1)
    bvh2 = bvh.bvh_create(xyz2)
    tcre = perf_counter() - tcre

    N = 10
    for i in range(N):

        pos1 = hm.rand_xform(cart_sd=0.7)
        pos2 = hm.rand_xform(cart_sd=0.7)

        tbvh = perf_counter()
        clash1 = bvh.bvh_isect(
            bvh1=bvh1, bvh2=bvh2, pos1=pos1, pos2=pos2, mindist=mindist
        )
        tbvh = perf_counter() - tbvh

        tn = perf_counter()
        clash2 = bvh.naive_isect(bvh1, bvh2, pos1, pos2, mindist)
        tn = perf_counter() - tn

        assert clash1 == clash2

        print(f"{i:3} clash {clash1:1} {tn / tbvh:8.2f}, {tn:1.6f}, {tbvh:1.6f}")

        totbvh += tbvh
        totnaive += tn

    print(
        f"iscet {N:,} iter bvh: {int(N/totbvh):,}/s fastnaive {int(N/totnaive):,}/s",
        f"ratio {int(totnaive/totbvh):,}x",
    )


def test_bvhf_min_cpp():
    assert bvh_test.test_bvh_test_min()


def test_bvh_min_dist_fixed():
    xyz1 = np.random.rand(5000, 3) + [0.9, 0.9, 0.0]
    xyz2 = np.random.rand(5000, 3)
    tcre = perf_counter()
    bvh1 = bvh.bvh_create(xyz1)
    bvh2 = bvh.bvh_create(xyz2)
    tcre = perf_counter() - tcre

    tbvh = perf_counter()
    d, i1, i2 = bvh.bvh_min_dist_fixed(bvh1, bvh2)
    tbvh = perf_counter() - tbvh
    dtest = np.linalg.norm(xyz1[i1] - xyz2[i2])
    assert np.allclose(d, dtest, atol=1e-6)

    # tnp = perf_counter()
    # dnp = np.min(np.linalg.norm(xyz1[:, None] - xyz2[None], axis=2))
    # tnp = perf_counter() - tnp

    tn = perf_counter()
    dn = bvh.naive_min_dist_fixed(bvh1, bvh2)
    tn = perf_counter() - tn

    print()
    print("from bvh:  ", d)
    print("from naive:", dn)
    assert np.allclose(dn, d, atol=1e-6)

    print(f"tnaivecpp {tn:5f} tbvh {tbvh:5f} tbvhcreate {tcre:5f}")
    print("bvh acceleration vs naive", tn / tbvh)
    assert tn / tbvh > 100


def test_bvh_min_dist():

    xyz1 = np.random.rand(1000, 3) - [0.5, 0.5, 0.5]
    xyz2 = np.random.rand(1000, 3) - [0.5, 0.5, 0.5]
    tcre = perf_counter()
    bvh1 = bvh.bvh_create(xyz1)
    bvh2 = bvh.bvh_create(xyz2)
    tcre = perf_counter() - tcre
    print()
    totbvh, totnaive = 0, 0
    N = 10
    for i in range(N):
        pos1 = hm.rand_xform(cart_sd=1)
        pos2 = hm.rand_xform(cart_sd=1)

        tbvh = perf_counter()
        d, i1, i2 = bvh.bvh_min_dist(bvh1, bvh2, pos1, pos2)
        tbvh = perf_counter() - tbvh
        dtest = np.linalg.norm(pos1 @ hm.hpoint(xyz1[i1]) - pos2 @ hm.hpoint(xyz2[i2]))
        assert np.allclose(d, dtest, atol=1e-6)

        tn = perf_counter()
        dn = bvh.naive_min_dist(bvh1, bvh2, pos1, pos2)
        tn = perf_counter() - tn
        assert np.allclose(dn, d, atol=1e-6)

        print(
            f"tnaivecpp {tn:1.6f} tbvh {tbvh:1.6f} tcpp/tbvh {tn/tbvh:8.1f}",
            np.linalg.norm(pos1[:3, 3]),
            dtest - d,
        )
        totnaive += tn
        totbvh += tbvh

    print(
        "total times",
        totbvh / N * 1000,
        "ms",
        totnaive / totbvh,
        totnaive,
        f"tcre {tcre:2.4f}",
    )


def test_bvh_slide_single_inline():

    bvh1 = bvh.bvh_create([[-10, 0, 0]])
    bvh2 = bvh.bvh_create([[0, 0, 0]])
    d = bvh.bvh_slide(bvh1, bvh2, np.eye(4), np.eye(4), rad=1.0, dirn=[1, 0, 0])
    assert d == 8
    # moves xyz1 to -2,0,0

    # should always come in from "infinity" from -direction
    bvh1 = bvh.bvh_create([[10, 0, 0]])
    bvh2 = bvh.bvh_create([[0, 0, 0]])
    d = bvh.bvh_slide(bvh1, bvh2, np.eye(4), np.eye(4), rad=1.0, dirn=[1, 0, 0])
    assert d == -12
    # also moves xyz1 to -2,0,0

    for i in range(100):
        np.random.seed(i)
        dirn = np.array([np.random.randn(), 0, 0])
        dirn /= np.linalg.norm(dirn)
        rad = np.abs(np.random.randn() / 10)
        xyz1 = np.array([[np.random.randn(), 0, 0]])
        xyz2 = np.array([[np.random.randn(), 0, 0]])
        bvh1 = bvh.bvh_create(xyz1)
        bvh2 = bvh.bvh_create(xyz2)
        d = bvh.bvh_slide(bvh1, bvh2, np.eye(4), np.eye(4), rad=rad, dirn=dirn)
        xyz1 += d * dirn
        assert np.allclose(np.linalg.norm(xyz1 - xyz2), 2 * rad, atol=1e-4)


def test_bvh_slide_single():

    nmiss = 0
    for i in range(100):
        # np.random.seed(i)
        dirn = np.random.randn(3)
        dirn /= np.linalg.norm(dirn)
        rad = np.abs(np.random.randn())
        xyz1 = np.random.randn(1, 3)
        xyz2 = np.random.randn(1, 3)
        bvh1 = bvh.bvh_create(xyz1)
        bvh2 = bvh.bvh_create(xyz2)
        d = bvh.bvh_slide(bvh1, bvh2, np.eye(4), np.eye(4), rad=rad, dirn=dirn)
        if d < 9e8:
            xyz1 += d * dirn
            assert np.allclose(np.linalg.norm(xyz1 - xyz2), 2 * rad, atol=1e-4)
        else:
            nmiss += 1
            delta = xyz2 - xyz1
            d0 = delta.dot(dirn)
            dperp2 = np.sum(delta * delta) - d0 * d0
            target_d2 = 4 * rad ** 2
            assert target_d2 < dperp2
    print("nmiss", nmiss, nmiss / 1000)


def test_bvh_slide_single_xform():

    nmiss = 0
    for i in range(1000):
        dirn = np.random.randn(3)
        dirn /= np.linalg.norm(dirn)
        rad = np.abs(np.random.randn() * 2.0)
        xyz1 = np.random.randn(1, 3)
        xyz2 = np.random.randn(1, 3)
        bvh1 = bvh.bvh_create(xyz1)
        bvh2 = bvh.bvh_create(xyz2)
        pos1 = hm.rand_xform()
        pos2 = hm.rand_xform()
        d = bvh.bvh_slide(bvh1, bvh2, pos1, pos2, rad=rad, dirn=dirn)
        if d < 9e8:
            p1 = (pos1 @ hm.hpoint(xyz1[0]))[:3] + d * dirn
            p2 = (pos2 @ hm.hpoint(xyz2[0]))[:3]
            assert np.allclose(np.linalg.norm(p1 - p2), 2 * rad, atol=1e-4)
        else:
            nmiss += 1
            p2 = pos2 @ hm.hpoint(xyz2[0])
            p1 = pos1 @ hm.hpoint(xyz1[0])
            delta = p2 - p1
            d0 = delta[:3].dot(dirn)
            dperp2 = np.sum(delta * delta) - d0 * d0
            target_d2 = 4 * rad ** 2
            assert target_d2 < dperp2
    print("nmiss", nmiss, nmiss / 1000)


def test_bvh_slide_whole():

    # timings wtih -Ofast
    # slide test 10,000 iter bvhslide float: 16,934/s double: 16,491/s bvhmin 17,968/s fracmiss: 0.0834

    # np.random.seed(0)
    N1, N2 = 1, 100
    totbvh, totbvhf, totmin = 0, 0, 0
    nmiss = 0
    for j in range(N1):
        xyz1 = np.random.rand(5000, 3) - [0.5, 0.5, 0.5]
        xyz2 = np.random.rand(5000, 3) - [0.5, 0.5, 0.5]
        # tcre = perf_counter()
        bvh1 = bvh.bvh_create(xyz1)
        bvh2 = bvh.bvh_create(xyz2)
        bvh1f = bvh.bvh_create_32bit(xyz1)
        bvh2f = bvh.bvh_create_32bit(xyz2)
        # tcre = perf_counter() - tcre
        for i in range(N2):
            dirn = np.random.randn(3)
            dirn /= np.linalg.norm(dirn)
            radius = 0.001 + np.random.rand() / 10
            pos1 = hm.rand_xform(cart_sd=0.5)
            pos2 = hm.rand_xform(cart_sd=0.5)

            tbvh = perf_counter()
            dslide = bvh.bvh_slide(bvh1, bvh2, pos1, pos2, radius, dirn)
            tbvh = perf_counter() - tbvh
            tbvhf = perf_counter()
            # dslide = bvh.bvh_slide_32bit(bvh1f, bvh2f, pos1, pos2, radius, dirn)
            tbvhf = perf_counter() - tbvhf

            if dslide > 9e8:
                tn = perf_counter()
                dn, i, j = bvh.bvh_min_dist(bvh1, bvh2, pos1, pos2)
                tn = perf_counter() - tn
                assert dn > 2 * radius
                nmiss += 1
            else:
                pos1 = hm.htrans(dirn * dslide) @ pos1
                tn = perf_counter()
                dn, i, j = bvh.bvh_min_dist(bvh1, bvh2, pos1, pos2)
                tn = perf_counter() - tn
                if not np.allclose(dn, 2 * radius, atol=1e-6):
                    print(dn, 2 * radius)
                assert np.allclose(dn, 2 * radius, atol=1e-6)

            # print(
            # i,
            # f"tnaivecpp {tn:1.6f} tbvh {tbvh:1.6f} tcpp/tbvh {tn/tbvh:8.1f}",
            # np.linalg.norm(pos1[:3, 3]),
            # dslide,
            # )
            totmin += tn
            totbvh += tbvh
            totbvhf += tbvhf
    N = N1 * N2
    print(
        f"slide test {N:,} iter bvhslide double: {int(N/totbvh):,}/s bvhmin {int(N/totmin):,}/s",
        # f"slide test {N:,} iter bvhslide float: {int(N/totbvhf):,}/s double: {int(N/totbvh):,}/s bvhmin {int(N/totmin):,}/s",
        f"fracmiss: {nmiss/N}",
    )


if __name__ == "__main__":
    # test_bvh_min_dist()
    # test_bvh_isect()
    test_bvh_slide_whole()
