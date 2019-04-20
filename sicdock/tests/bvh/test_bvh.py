from time import perf_counter
import numpy as np
from cppimport import import_hook

# import cppimport

# cppimport.set_quiet(False)

from sicdock.bvh import bvh_test
from sicdock.bvh import bvh
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
        assert len(bvh1) == 1000

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
    # assert tn / tbvh > 100


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


def test_bvh_min_dist_floormin():

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
    N1, N2 = 2, 10
    totbvh, totbvhf, totmin = 0, 0, 0
    nmiss = 0
    for j in range(N1):
        xyz1 = np.random.rand(5000, 3) - [0.5, 0.5, 0.5]
        xyz2 = np.random.rand(5000, 3) - [0.5, 0.5, 0.5]
        # tcre = perf_counter()
        bvh1 = bvh.bvh_create(xyz1)
        bvh2 = bvh.bvh_create(xyz2)
        # bvh1f = bvh.bvh_create_32bit(xyz1)
        # bvh2f = bvh.bvh_create_32bit(xyz2)
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


def test_collect_pairs_simple():
    print("test_collect_pairs_simple")
    bufbvh = -np.ones((100, 2), dtype="i4")
    bufnai = -np.ones((100, 2), dtype="i4")
    bvh1 = bvh.bvh_create([[0, 0, 0], [0, 2, 0]])
    bvh2 = bvh.bvh_create([[0.9, 0, 0], [0.9, 2, 0]])
    assert len(bvh1) == 2
    mindist = 1.0

    pos1 = np.eye(4)
    pos2 = np.eye(4)
    nbvh = bvh.bvh_collect_pairs(bvh1, bvh2, pos1, pos2, mindist, bufbvh)
    nnai = bvh.naive_collect_pairs(bvh1, bvh2, pos1, pos2, mindist, bufnai)
    assert nbvh == 2 and nnai == 2
    assert np.all(bufbvh[:nbvh] == [[0, 0], [1, 1]])
    assert np.all(bufnai[:nnai] == [[0, 0], [1, 1]])

    pos1 = hm.htrans([0, 2, 0])
    nbvh = bvh.bvh_collect_pairs(bvh1, bvh2, pos1, pos2, mindist, bufbvh)
    nnai = bvh.naive_collect_pairs(bvh1, bvh2, pos1, pos2, mindist, bufnai)
    assert nbvh == 1 and nnai == 1
    assert np.all(bufbvh[:nbvh] == [[0, 1]])
    assert np.all(bufnai[:nnai] == [[0, 1]])

    pos1 = hm.htrans([0, -2, 0])
    nbvh = bvh.bvh_collect_pairs(bvh1, bvh2, pos1, pos2, mindist, bufbvh)
    nnai = bvh.naive_collect_pairs(bvh1, bvh2, pos1, pos2, mindist, bufnai)
    assert nbvh == 1 and nnai == 1
    assert np.all(bufbvh[:nbvh] == [[1, 0]])
    assert np.all(bufnai[:nnai] == [[1, 0]])


def test_collect_pairs_simple_selection():
    print("test_collect_pairs_simple_selection")
    bufbvh = -np.ones((100, 2), dtype="i4")
    bufnai = -np.ones((100, 2), dtype="i4")
    crd1 = [[0, 0, 0], [0, 0, 0], [0, 2, 0], [0, 0, 0]]
    crd2 = [[0, 0, 0], [0.9, 0, 0], [0, 0, 0], [0.9, 2, 0]]
    mask1 = [1, 0, 1, 0]
    mask2 = [0, 1, 0, 1]
    bvh1 = bvh.bvh_create(crd1, mask1)
    bvh2 = bvh.bvh_create(crd2, mask2)
    assert len(bvh1) == 2
    assert np.allclose(bvh1.radius(), 1.0, atol=1e-6)
    assert np.allclose(bvh1.center(), [0, 1, 0], atol=1e-6)
    mindist = 1.0

    pos1 = np.eye(4)
    pos2 = np.eye(4)
    nbvh = bvh.bvh_collect_pairs(bvh1, bvh2, pos1, pos2, mindist, bufbvh)
    nnai = bvh.naive_collect_pairs(bvh1, bvh2, pos1, pos2, mindist, bufnai)
    assert nbvh == 2 and nnai == 2
    assert np.all(bufbvh[:nbvh] == [[0, 1], [2, 3]])
    assert np.all(bufnai[:nnai] == [[0, 1], [2, 3]])

    pos1 = hm.htrans([0, 2, 0])
    nbvh = bvh.bvh_collect_pairs(bvh1, bvh2, pos1, pos2, mindist, bufbvh)
    nnai = bvh.naive_collect_pairs(bvh1, bvh2, pos1, pos2, mindist, bufnai)
    assert nbvh == 1 and nnai == 1
    assert np.all(bufbvh[:nbvh] == [[0, 3]])
    assert np.all(bufnai[:nnai] == [[0, 3]])

    pos1 = hm.htrans([0, -2, 0])
    nbvh = bvh.bvh_collect_pairs(bvh1, bvh2, pos1, pos2, mindist, bufbvh)
    nnai = bvh.naive_collect_pairs(bvh1, bvh2, pos1, pos2, mindist, bufnai)
    assert nbvh == 1 and nnai == 1
    assert np.all(bufbvh[:nbvh] == [[2, 1]])
    assert np.all(bufnai[:nnai] == [[2, 1]])


def test_collect_pairs():
    N1, N2 = 2, 10
    N = N1 * N2
    Npts = 1000
    totbvh, totbvhf, totmin = 0, 0, 0
    totbvh, totnai, totct, ntot = 0, 0, 0, 0
    bufbvh = -np.ones((Npts * Npts, 2), dtype="i4")
    bufnai = -np.ones((Npts * Npts, 2), dtype="i4")
    for j in range(N1):
        xyz1 = np.random.rand(Npts, 3) - [0.5, 0.5, 0.5]
        xyz2 = np.random.rand(Npts, 3) - [0.5, 0.5, 0.5]
        bvh1 = bvh.bvh_create(xyz1)
        bvh2 = bvh.bvh_create(xyz2)
        for i in range(N2):
            mindist = 0.002 + np.random.rand() / 10
            while 1:
                pos1 = hm.rand_xform(cart_sd=0.5)
                pos2 = hm.rand_xform(cart_sd=0.5)
                d = np.linalg.norm(pos1[:, 3] - pos2[:, 3])
                if 0.8 < d < 1.3:
                    break

            tbvh = perf_counter()
            nbvh = bvh.bvh_collect_pairs(bvh1, bvh2, pos1, pos2, mindist, bufbvh)
            tbvh = perf_counter() - tbvh

            tnai = perf_counter()
            nnai = bvh.naive_collect_pairs(bvh1, bvh2, pos1, pos2, mindist, bufnai)
            tnai = perf_counter() - tnai

            tct = perf_counter()
            nct = bvh.bvh_count_pairs(bvh1, bvh2, pos1, pos2, mindist)
            tct = perf_counter() - tct
            ntot += nct

            assert nct == nbvh
            totnai += 1

            totbvh += tbvh
            totnai += tnai
            totct += tct

            assert nbvh == nnai
            if nbvh == 0:
                continue

            o = np.lexsort((bufbvh[:nbvh, 1], bufbvh[:nbvh, 0]))
            bufbvh[:nbvh] = bufbvh[:nbvh][o]
            o = np.lexsort((bufnai[:nnai, 1], bufnai[:nnai, 0]))
            bufnai[:nnai] = bufnai[:nnai][o]
            assert np.all(bufbvh[:nbvh] == bufnai[:nnai])

            pair1 = pos1 @ hm.hpoint(xyz1[bufbvh[:nbvh, 0]])[..., None]
            pair2 = pos2 @ hm.hpoint(xyz2[bufbvh[:nbvh, 1]])[..., None]
            dpair = np.linalg.norm(pair2 - pair1, axis=1)
            assert np.max(dpair) <= mindist

    print(
        f"collect test {N:,} iter bvh {int(N/totbvh):,}/s naive {int(N/totnai):,}/s ratio {totnai/totbvh:7.2f} count-only {int(N/totct):,}/s avg cnt {ntot/N}"
    )


def test_slide_collect_pairs():

    # timings wtih -Ofast
    # slide test 10,000 iter bvhslide float: 16,934/s double: 16,491/s bvhmin 17,968/s fracmiss: 0.0834

    # np.random.seed(0)
    N1, N2 = 2, 50
    Npts = 5000
    totbvh, totbvhf, totcol, totmin = 0, 0, 0, 0
    nhit = 0
    buf = -np.ones((Npts * Npts, 2), dtype="i4")
    for j in range(N1):
        xyz1 = np.random.rand(Npts, 3) - [0.5, 0.5, 0.5]
        xyz2 = np.random.rand(Npts, 3) - [0.5, 0.5, 0.5]
        xyzcol1 = xyz1[: int(Npts / 5)]
        xyzcol2 = xyz2[: int(Npts / 5)]
        # tcre = perf_counter()
        bvh1 = bvh.bvh_create(xyz1)
        bvh2 = bvh.bvh_create(xyz2)
        bvhcol1 = bvh.bvh_create(xyzcol1)
        bvhcol2 = bvh.bvh_create(xyzcol2)
        # tcre = perf_counter() - tcre
        for i in range(N2):
            dirn = np.random.randn(3)
            dirn /= np.linalg.norm(dirn)
            radius = 0.001 + np.random.rand() / 10
            pairdis = 3 * radius
            pos1 = hm.rand_xform(cart_sd=0.5)
            pos2 = hm.rand_xform(cart_sd=0.5)

            tbvh = perf_counter()
            dslide = bvh.bvh_slide(bvh1, bvh2, pos1, pos2, radius, dirn)
            tbvh = perf_counter() - tbvh

            if dslide > 9e8:
                tn = perf_counter()
                dn, i, j = bvh.bvh_min_dist(bvh1, bvh2, pos1, pos2)
                tn = perf_counter() - tn
                assert dn > 2 * radius
            else:
                nhit += 1
                pos1 = hm.htrans(dirn * dslide) @ pos1
                tn = perf_counter()
                dn, i, j = bvh.bvh_min_dist(bvh1, bvh2, pos1, pos2)
                tn = perf_counter() - tn
                if not np.allclose(dn, 2 * radius, atol=1e-6):
                    print(dn, 2 * radius)
                assert np.allclose(dn, 2 * radius, atol=1e-6)

                tcol = perf_counter()
                npair = bvh.bvh_collect_pairs(
                    bvhcol1, bvhcol2, pos1, pos2, pairdis, buf
                )
                if npair > 0:
                    tcol = perf_counter() - tcol
                    totcol += tcol
                    pair1 = pos1 @ hm.hpoint(xyzcol1[buf[:npair, 0]])[..., None]
                    pair2 = pos2 @ hm.hpoint(xyzcol2[buf[:npair, 1]])[..., None]
                    dpair = np.linalg.norm(pair2 - pair1, axis=1)
                    assert np.max(dpair) <= pairdis

            totmin += tn
            totbvh += tbvh

    N = N1 * N2
    print(
        f"slide test {N:,} iter bvhslide double: {int(N/totbvh):,}/s bvhmin {int(N/totmin):,}/s",
        # f"slide test {N:,} iter bvhslide float: {int(N/totbvhf):,}/s double: {int(N/totbvh):,}/s bvhmin {int(N/totmin):,}/s",
        f"fracmiss: {nhit/N} collect {int(nhit/totcol):,}/s",
    )


def test_bvh_accessors():
    xyz = np.random.rand(10, 3) - [0.5, 0.5, 0.5]
    b = bvh.bvh_create(xyz)
    assert np.allclose(b.com()[:3], np.mean(xyz, axis=0))
    p = b.centers()
    dmat = np.linalg.norm(p[:, :3] - xyz[:, None], axis=2)
    assert np.allclose(np.min(dmat, axis=1), 0)


if __name__ == "__main__":
    # test_bvh_min_dist()
    # test_bvh_isect()
    # test_bvh_slide_whole()
    # test_collect_pairs_simple()
    # test_collect_pairs_simple_selection()
    # test_collect_pairs()
    # test_slide_collect_pairs()
    test_bvh_accessors()