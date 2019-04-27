from time import perf_counter
import itertools as it
import numpy as np
from cppimport import import_hook
from sicdock.sampling.xform_hierarchy import *
from sicdock.bvh import bvh_nd
import homog as hm
from scipy.spatial.distance import cdist


def urange(*args):
    return np.arange(*args, dtype="u8")


def urandint(*args):
    return np.random.randint(*args, dtype="u8")


def test_cart_hier1():
    ch = CartHier1D([0], [1], [1])
    for resl in range(1, 10):
        i, t = ch.get_trans(resl, np.arange(ch.size(resl), dtype="u8"))
        assert len(i) == 2 ** resl == ch.size(resl)
        diff = np.diff(t, axis=0)
        assert np.min(diff) == np.max(diff)
        assert np.allclose(np.min(diff), 1 / 2 ** resl)
    for i in range(10):
        tmp = np.random.randn(2)
        lb, ub = min(tmp), max(tmp)
        bs = urandint(1, 11)
        ch2 = CartHier1D([lb], [ub], [bs])
        for resl in range(4):
            assert ch2.size(resl) == ch.size(resl) * bs


def test_xform_hierarchy_product():
    oh = OriHier(999)
    ch = CartHier3D([0] * 3, [1] * 3, [1] * 3)
    xh = XformHier([0] * 3, [1] * 3, [1] * 3, 999)
    resl = 0
    i, x = xh.get_xforms(resl, urange(xh.size(resl)))
    i, o = oh.get_ori(resl, urange(oh.size(resl)))
    i, t = ch.get_trans(resl, urange(ch.size(resl)))
    assert np.allclose(x[:, :3, :3], o)
    assert np.allclose(x[:, :3, 3], t)

    resl = 1
    i, x = xh.get_xforms(resl, urange(0, xh.size(resl), 8))
    i, o = oh.get_ori(resl, urange(oh.size(resl)))
    i, t = ch.get_trans(resl, urange(ch.size(resl)))
    assert np.allclose(x.reshape(24, -1, 4, 4)[0, :, :3, 3], t)
    assert np.allclose(
        x.reshape(24, -1, 4, 4)[:, ::8, :3, :3], o.reshape(24, -1, 3, 3)[:, ::8]
    )


def test_xform_hierarchy_product_zorder():
    for ang in [999, 30, 20, 10, 5]:
        ho = OriHier(ang)
        hx = XformHier([0] * 3, [1] * 3, [1] * 3, ang)
        assert hx.ori_nside() == ho.ori_nside()
        Nmax = 1_000
        for resl in range(8):
            n0 = ho.size(resl)
            idx0 = urange(n0) if n0 < Nmax else urandint(0, n0, Nmax)
            io, mo = ho.get_ori(resl, idx0)
            z6 = np.zeros((np.sum(io), 7), dtype="u8")
            z6[:, :4] = zorder3coeffs(idx0[io], resl)
            ix = coeffs6zorder(z6, resl)
            wx, xx = hx.get_xforms(resl, ix)
            assert len(wx) == len(ix)
            assert np.allclose(xx[:, :3, :3], mo)

    for i in range(10):
        tmp = np.random.randn(2, 3)
        lb = np.minimum(*tmp)
        ub = np.maximum(*tmp)
        bs = urandint(1, 10, 3)
        hc = CartHier3D(lb, ub, bs)
        hx = XformHier(lb, ub, bs, 999)
        Nmax = 10_000
        for resl in range(8):
            n0 = ho.size(resl)
            idx0 = urange(n0) if n0 < Nmax else urandint(0, n0, Nmax)
            io, to = hc.get_trans(resl, idx0)
            z6 = np.zeros((np.sum(io), 7), dtype="u8")
            z3 = zorder3coeffs(idx0[io], resl)
            z6[:, 0] = 24 * z3[:, 0]
            z6[:, 4:] = z3[:, 1:]
            ix = coeffs6zorder(z6, resl)
            wx, xx = hx.get_xforms(resl, ix)
            # assert len(wx) == len(ix)
            assert np.allclose(xx[:, :3, 3], to[wx])


def test_xform_hierarchy_ctor():
    xh = XformHier(lb=[0, 0, 0], ub=[2, 2, 2], bs=[2, 2, 2], ori_resl=999.0)


def test_xform_hierarchy_get_xforms():
    for a, b, c in it.product([1, 2], [1, 2], [1, 2]):
        xh = XformHier(lb=[0, 0, 0], ub=[a, b, c], bs=[1, 1, 1], ori_resl=999.0)
        idx, xform = xh.get_xforms(0, np.arange(10, dtype="u8"))
        assert np.allclose(xform[:, :3, 3], [a * 0.5, b * 0.5, c * 0.5])

        idx, xform = xh.get_xforms(1, urange(64))
        assert np.all(idx)
        t = xform[:, :3, 3]
        assert np.all(
            np.unique(t, axis=0)
            == [
                [a * 0.25, b * 0.25, c * 0.25],
                [a * 0.25, b * 0.25, c * 0.75],
                [a * 0.25, b * 0.75, c * 0.25],
                [a * 0.25, b * 0.75, c * 0.75],
                [a * 0.75, b * 0.25, c * 0.25],
                [a * 0.75, b * 0.25, c * 0.75],
                [a * 0.75, b * 0.75, c * 0.25],
                [a * 0.75, b * 0.75, c * 0.75],
            ]
        )
    xh = XformHier(lb=[-1, -1, -1], ub=[0, 0, 0], bs=[1, 1, 1], ori_resl=999.0)
    idx, xform = xh.get_xforms(2, np.arange(64, dtype="u8"))
    t = np.unique(xform[:, :3, 3], axis=0)
    assert np.all(
        t
        == [
            [-0.875, -0.875, -0.875],
            [-0.875, -0.875, -0.625],
            [-0.875, -0.625, -0.875],
            [-0.875, -0.625, -0.625],
            [-0.625, -0.875, -0.875],
            [-0.625, -0.875, -0.625],
            [-0.625, -0.625, -0.875],
            [-0.625, -0.625, -0.625],
        ]
    )


def test_xform_hierarchy_get_xforms_bs():
    xh = XformHier(lb=[0, 0, 0], ub=[4, 4, 4], bs=[2, 2, 2], ori_resl=999.0)
    idx, xform = xh.get_xforms(0, np.arange(xh.size(0), dtype="u8"))
    t = xform[:, :3, 3]
    u = np.unique(t, axis=0)
    assert np.all(
        u
        == [
            [1.0, 1.0, 1.0],
            [1.0, 1.0, 3.0],
            [1.0, 3.0, 1.0],
            [1.0, 3.0, 3.0],
            [3.0, 1.0, 1.0],
            [3.0, 1.0, 3.0],
            [3.0, 3.0, 1.0],
            [3.0, 3.0, 3.0],
        ]
    )
    for a, b, c in it.product([1, 2], [1, 2], [1, 2]):
        xh = XformHier(lb=[0, 0, 0], ub=[a, b, c], bs=[a, b, c], ori_resl=999.0)
        idx, xform = xh.get_xforms(0, np.arange(xh.size(0), dtype="u8"))
        t = xform[:, :3, 3]
        u = np.unique(t, axis=0)
        print(u)
        print(np.sum(u[:, 0] == 0.5), a, b, c)
        # assert np.sum(u[:, 0] == 0.5) == b * c


def test_xform_hierarchy_expand_top_N():
    xh = XformHier(lb=[0, 0, 0], ub=[2, 2, 2], bs=[2, 2, 2], ori_resl=30.0)
    scoreindex = np.empty(10, dtype=[("score", "f8"), ("index", "u8")])
    scoreindex["index"] = np.arange(10)
    scoreindex["score"] = np.arange(10)
    idx1, xform1 = xh.expand_top_N(3, 0, scoreindex)

    score = np.arange(10).astype("f8")
    index = np.arange(10).astype("u8")
    idx2, xform2 = xh.expand_top_N(3, 0, score, index)

    assert np.all(idx1 == idx2)
    assert np.allclose(xform1, xform2)

    idx1.sort()
    assert np.all(idx1 == np.arange(7 * 64, 10 * 64))
    idx2, xform2 = xh.expand_top_N(3, 0, -score, index)
    idx2.sort()
    assert np.all(idx2 == np.arange(3 * 64))


def test_zorder():
    idx = urange(1e5)
    for resl in range(5):
        coef = zorder6coeffs(idx, resl)
        assert np.all(coeffs6zorder(coef, resl) == idx)

    for resl in range(10):
        n = min(1_000_000, 100 * 2 ** resl)
        coef = urandint(0, 2 ** resl, (n, 7))
        coef[:, 0] = urandint(0, 1024, n)
        idx = coeffs6zorder(coef, resl)
        coef2 = zorder6coeffs(idx, resl)
        # print(idx[:3])
        # print(coef[:3])
        assert np.all(coef2 == coef)


def test_ori_hier_all2():
    minrange = np.array(
        [(116.5, 116.6), (68.8, 68.9), (39.4, 42.5), (19.3, 20.7), (9.5, 9.9), (5, 6)]
    )
    corner = [(0, 0), (0, 0), (0.125, 14)]
    ohier = OriHier(9e9)
    for resl in range(3):
        w, o = ohier.get_ori(resl, urange(ohier.size(resl)))
        assert np.allclose(np.linalg.det(o), 1)
        rel = o.swapaxes(1, 2)[:, None] @ o
        amat = hm.angle_of(rel)
        assert np.allclose(amat.diagonal(), 0)
        np.fill_diagonal(amat, 9e9)
        mn = amat.min(axis=0)
        cfrac, cang = corner[resl]
        print(np.unique(mn), cang, cfrac)
        assert np.sum(mn < cang * np.pi / 180) / len(mn) == cfrac
        mn = mn[mn > cang]
        # print(resl, len(mn), np.unique(mn) * 180 / np.pi)
        lb, ub = minrange[resl] / 180 * np.pi

        assert np.all(lb < mn)
        assert np.all(mn < ub)


def test_ori_hier_1cell():
    minrange = np.array(
        [(116.5, 116.6), (73.6, 73.7), (39.4, 42.5), (19.3, 23.0), (9.5, 12)]
    )
    ohier = OriHier(9e9)
    for resl in range(1, 4):
        w, o = ohier.get_ori(resl, urange(ohier.size(resl) / 24))
        assert np.allclose(np.linalg.det(o), 1)
        rel = o.swapaxes(1, 2)[:, None] @ o
        amat = hm.angle_of(rel)
        assert np.allclose(amat.diagonal(), 0)
        np.fill_diagonal(amat, 9e9)
        mn = amat.min(axis=0)
        # print(resl, len(mn), np.unique(mn) * 180 / np.pi)
        lb, ub = minrange[resl] / 180 * np.pi
        print(resl, np.unique((mn * 180 / np.pi).round(1)))
        assert np.all(lb < mn)
        assert np.all(mn < ub)


def test_ori_hier_rand():
    # r1 = hm.rand_xform(100)[:, :3, :3]
    # r2 = hm.rand_xform(100)[:, :3, :3]
    # q1 = hm.quat.rot_to_quat(r1)
    # q2 = hm.quat.rot_to_quat(r2)
    # qdist = 3 * np.minimum(cdist(q1, q2), cdist(q1, -q2))
    # # qdist = np.arccos(2 * np.sum(q1[:, None] * q2, axis=2) - 1)
    # adist = hm.angle_of(r1[:, None].swapaxes(-1, -2) @ r2)
    # import matplotlib.pyplot as plt
    # plt.scatter(qdist, adist)
    # plt.show()
    # return
    N = 10_000
    ohier = OriHier(9e9)
    cut = [99, 59, 31, 16, 8, 4]  # not quite angles!!
    for resl in range(4):
        w, o = ohier.get_ori(resl, urange(ohier.size(resl)))
        assert np.allclose(np.linalg.det(o), 1)
        bvh_ohier = bvh_nd.create_bvh_quat(o.reshape(-1, 9).copy())
        dis = np.empty(N)
        samp = hm.rand_xform(N)[:, :3, :3]
        for j in range(N):
            dis[j] = bvh_nd.bvh_min_one_quat(bvh_ohier, samp[j])
        mx = 3.0 * np.max(dis) * 180 / np.pi
        me = 3.0 * np.mean(dis) * 180 / np.pi
        print(f"{resl} {len(bvh_ohier):6,} {mx} {me} {mx / me}")
        assert mx < cut[resl]


def test_avg_dist():
    from sicdock.bvh import bvh

    N = 1000
    ch = CartHier3D([-1, -1, -1], [2, 2, 2], [1, 1, 1])
    for resl in range(1, 4):
        i, grid = ch.get_trans(resl, urange(ch.size(3)))
        gridbvh = bvh.bvh_create(grid)
        dis = np.empty(N)
        samp = np.random.rand(N, 3)
        for j in range(N):
            dis[j] = bvh.bvh_min_dist_one(gridbvh, samp[j])[0]
        mx = 3.0 * np.max(dis) * 180 / np.pi
        me = 3.0 * np.mean(dis) * 180 / np.pi
        print(f"{resl} {len(gridbvh):6,} {mx} {me} {mx / me}")
        assert mx / me < 3


if __name__ == "__main__":
    # test_zorder()
    # test_cart_hier1()
    # test_xform_hierarchy_product()
    # test_xform_hierarchy_product_zorder()
    # test_xform_hierarchy_ctor()
    # test_xform_hierarchy_get_xforms()
    # test_xform_hierarchy_get_xforms_bs()
    # test_xform_hierarchy_expand_top_N()
    # test_ori_hier_all2()
    # test_ori_hier_1cell()
    test_ori_hier_rand()
    # test_avg_dist()
