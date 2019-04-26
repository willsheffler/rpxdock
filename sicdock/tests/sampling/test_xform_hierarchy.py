import itertools as it
import numpy as np
from cppimport import import_hook
from sicdock.sampling.xform_hierarchy import *


def test_xform_hierarchy_cpp():
    assert TEST_xform_hier_simple()


def test_xform_hierarchy_ctor():
    xh = XformHier(lb=[0, 0, 0], ub=[2, 2, 2], bs=[2, 2, 2], ori_resl=999.0)


def test_xform_hierarchy_get_xforms():
    for a, b, c in it.product([1, 2], [1, 2], [1, 2]):
        xh = XformHier(lb=[0, 0, 0], ub=[a, b, c], bs=[1, 1, 1], ori_resl=999.0)
        idx, xform = xh.get_xforms(0, np.arange(10, dtype="u8"))
        assert np.allclose(xform[:, :3, 3], [a * 0.5, b * 0.5, c * 0.5])

        idx, xform = xh.get_xforms(1, np.arange(64, dtype="u8"))
        assert np.all(idx == np.arange(64))
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


def test_xform_hierarchy_expand_top_N_locality():
    xh = XformHier(lb=[0, 0, 0], ub=[2, 2, 2], bs=[2, 2, 2], ori_resl=30.0)


if __name__ == "__main__":
    # test_xform_hierarchy_cpp()
    # test_xform_hierarchy_ctor()
    # test_xform_hierarchy_get_xforms()
    test_xform_hierarchy_get_xforms_bs()
    # test_xform_hierarchy_expand_top_N()
    # test_xform_hierarchy_expand_top_N_locality()
