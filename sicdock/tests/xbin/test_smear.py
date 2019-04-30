from collections import Counter
import numpy as np
import homog as hm
from sicdock.geom import BCC6
from sicdock.xbin import XBin
from sicdock.xbin.smear import smear
from sicdock.phmap import PHMap_u8f8
from sicdock.geom.rotation import angle_of_3x3


def test_smear_no_midpoints():
    for r in range(1, 6):
        w = 2 * r + 1
        cart_resl = 1.0
        xb = XBin(cart_resl, 9e9)
        gr = xb.grid6
        pm = PHMap_u8f8()
        cen = np.eye(4)
        kcen = xb.key_of(np.eye(4))
        bcen = xb.bincen_of(kcen)
        assert np.allclose(np.eye(4), bcen)
        phm = PHMap_u8f8()
        phm[xb.key_of(bcen)] = 1.0
        smeared = smear(xb, phm, radius=r, midpoints=False, extra_midpoints=False)
        assert isinstance(smeared, PHMap_u8f8)
        assert len(smeared) == w ** 3
        k, v = smeared.items_array()
        x = xb.bincen_of(k)
        cart_dis = np.linalg.norm(bcen[0, :3, 3] - x[:, :3, 3], axis=1)
        assert np.min(cart_dis) == 0
        assert list(Counter(x[:, 0, 3]).values()) == ([w ** 2] * w)
        assert list(Counter(x[:, 1, 3]).values()) == ([w ** 2] * w)
        assert list(Counter(x[:, 2, 3]).values()) == ([w ** 2] * w)
        ori_dist = angle_of_3x3(x[:, :3, :3])
        assert np.max(ori_dist) == 0


def test_smear_midpoints():
    for r in range(1, 6):
        w = 2 * r + 1
        cart_resl = 1.0
        xb = XBin(cart_resl, 9e9)
        gr = xb.grid6
        pm = PHMap_u8f8()
        cen = np.eye(4)
        kcen = xb.key_of(np.eye(4))
        bcen = xb.bincen_of(kcen)
        assert np.allclose(np.eye(4), bcen)
        phm = PHMap_u8f8()
        phm[xb.key_of(bcen)] = 1.0
        smeared = smear(xb, phm, radius=r, midpoints=True, extra_midpoints=False)
        assert isinstance(smeared, PHMap_u8f8)
        assert len(smeared) == w ** 3 + (w - 1) ** 3
        k, v = smeared.items_array()
        x = xb.bincen_of(k)
        cart_dis = np.linalg.norm(bcen[0, :3, 3] - x[:, :3, 3], axis=1)
        assert np.min(cart_dis) == 0
        # print(sorted(Counter(x[:, 0, 3]).values()))
        counts = [(w - 1) ** 2] * (w - 1) + [w ** 2] * w
        assert sorted(Counter(x[:, 0, 3]).values()) == counts
        assert sorted(Counter(x[:, 1, 3]).values()) == counts
        assert sorted(Counter(x[:, 2, 3]).values()) == counts
        ori_dist = angle_of_3x3(x[:, :3, :3])
        assert np.allclose(np.unique(ori_dist), [0.0, 1.24466863])


def test_smear_midpoints_alldims():
    for r in range(1, 6):
        w = 2 * r + 1
        cart_resl = 1.0
        xb = XBin(cart_resl, 9e9)
        gr = xb.grid6
        pm = PHMap_u8f8()
        cen = np.eye(4)
        kcen = xb.key_of(np.eye(4))
        bcen = xb.bincen_of(kcen)
        assert np.allclose(np.eye(4), bcen)
        phm = PHMap_u8f8()
        phm[xb.key_of(bcen)] = 1.0
        smeared = smear(xb, phm, radius=r, midpoints=True, extra_midpoints=True)
        assert isinstance(smeared, PHMap_u8f8)
        assert len(smeared) == w ** 3 + 8 * (w - 1) ** 3
        k, v = smeared.items_array()
        x = xb.bincen_of(k)
        cart_dis = np.linalg.norm(bcen[0, :3, 3] - x[:, :3, 3], axis=1)
        d = 0.57787751
        uvals = np.arange(-2 * r, 2 * r + 0.001) * d
        assert np.allclose(np.unique(x[:, 0, 3]), uvals)
        assert np.allclose(np.unique(x[:, 1, 3]), uvals)
        assert np.allclose(np.unique(x[:, 2, 3]), uvals)
        counts = [w ** 2] * w + [8 * (w - 1) ** 2] * (w - 1)
        assert sorted(Counter(x[:, 0, 3]).values()) == counts
        assert sorted(Counter(x[:, 1, 3]).values()) == counts
        assert sorted(Counter(x[:, 2, 3]).values()) == counts
        ori_dist = angle_of_3x3(x[:, :3, :3])
        assert np.allclose(np.unique(ori_dist), [0.0, 1.24466863])


if __name__ == "__main__":
    # test_smear_no_midpoints()
    # test_smear_midpoints()
    test_smear_midpoints_alldims()
