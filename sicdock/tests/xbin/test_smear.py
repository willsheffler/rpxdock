from time import perf_counter
from collections import Counter
import numpy as np
import homog as hm
from sicdock.geom import BCC6
from sicdock.xbin import XBin
from sicdock.xbin.smear import smear
from sicdock.phmap import PHMap_u8f8
from sicdock.geom.rotation import angle_of_3x3

xident_f4 = np.eye(4).astype("f4")


def test_smear():
    for r in range(1, 6):
        w = 2 * r + 1
        cart_resl = 1.0
        xb = XBin(cart_resl, 9e9)
        gr = xb.grid6
        pm = PHMap_u8f8()
        cen = xident_f4
        kcen = xb.key_of(xident_f4)
        bcen = xb.bincen_of(kcen)
        assert np.allclose(cen, bcen, atol=1e-4)
        phm = PHMap_u8f8()
        phm[xb.key_of(bcen)] = 1.0
        smeared = smear(xb, phm, radius=r, extrahalf=0, oddlast3=0, sphere=0)
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


def test_smear_oddori():
    for r in range(1, 6):
        w = 2 * r + 1
        cart_resl = 1.0
        xb = XBin(cart_resl, 9e9)
        gr = xb.grid6
        pm = PHMap_u8f8()
        cen = xident_f4
        kcen = xb.key_of(xident_f4)
        bcen = xb.bincen_of(kcen)
        assert np.allclose(cen, bcen, atol=1e-4)
        phm = PHMap_u8f8()
        phm[xb.key_of(bcen)] = 1.0
        smeared = smear(xb, phm, radius=r, extrahalf=0, oddlast3=1, sphere=0)
        assert isinstance(smeared, PHMap_u8f8)
        assert len(smeared) == w ** 3 + 8 * (w - 1) ** 3
        k, v = smeared.items_array()
        x = xb.bincen_of(k)
        cart_dis = np.linalg.norm(bcen[0, :3, 3] - x[:, :3, 3], axis=1)
        d = 0.57787751
        uvals = np.arange(-2 * r, 2 * r + 0.001) * d
        assert np.allclose(np.unique(x[:, 0, 3]), uvals, atol=1e-4)
        assert np.allclose(np.unique(x[:, 1, 3]), uvals, atol=1e-4)
        assert np.allclose(np.unique(x[:, 2, 3]), uvals, atol=1e-4)
        counts = [w ** 2] * w + [8 * (w - 1) ** 2] * (w - 1)
        assert sorted(Counter(x[:, 0, 3]).values()) == counts
        assert sorted(Counter(x[:, 1, 3]).values()) == counts
        assert sorted(Counter(x[:, 2, 3]).values()) == counts
        ori_dist = angle_of_3x3(x[:, :3, :3])
        assert np.allclose(np.unique(ori_dist), [0.0, 1.24466863])


def test_smear_oddori_sphere():
    counts = [
        [5, 5, 9, 32, 32],
        [9, 9, 21, 21, 21, 96, 96, 128, 128],
        [9, 9, 25, 25, 37, 37, 37, 128, 128, 256, 256, 256, 256],
        [13, 13, 37, 37, 49, 49, 61, 61, 69, 192, 192, 352, 352, 416, 416, 480, 480],
        [21, 21, 45, 45, 69, 69, 89, 89, 97, 97, 97, 256, 256]
        + [416, 416, 608, 608, 704, 704, 704, 704],
    ]

    for r in range(1, 6):
        w = 2 * r + 1
        cart_resl = 1.0
        xb = XBin(cart_resl, 9e9)
        gr = xb.grid6
        pm = PHMap_u8f8()
        cen = xident_f4
        kcen = xb.key_of(xident_f4)
        bcen = xb.bincen_of(kcen)
        assert np.allclose(cen, bcen, atol=1e-4)
        phm = PHMap_u8f8()
        phm[xb.key_of(bcen)] = 1.0
        smeared = smear(xb, phm, radius=r, extrahalf=0, oddlast3=1, sphere=1)
        assert isinstance(smeared, PHMap_u8f8)
        assert len(smeared) == [83, 529, 1459, 3269, 6115][r - 1]

        k, v = smeared.items_array()
        x = xb.bincen_of(k)
        cart_dis = np.linalg.norm(bcen[0, :3, 3] - x[:, :3, 3], axis=1)
        d = 0.57787751
        uvals = np.arange(-2 * r, 2 * r + 0.001) * d
        assert np.allclose(np.unique(x[:, 0, 3]), uvals, atol=1e-4)
        assert np.allclose(np.unique(x[:, 1, 3]), uvals, atol=1e-4)
        assert np.allclose(np.unique(x[:, 2, 3]), uvals, atol=1e-4)
        assert sorted(Counter(x[:, 0, 3]).values()) == counts[r - 1]
        assert sorted(Counter(x[:, 1, 3]).values()) == counts[r - 1]
        assert sorted(Counter(x[:, 2, 3]).values()) == counts[r - 1]
        ori_dist = angle_of_3x3(x[:, :3, :3])
        assert np.allclose(np.unique(ori_dist), [0.0, 1.24466863])


def test_smear_exhalf_oddori_sphere():
    counts = [
        [5, 5, 9, 32, 32],
        [9, 9, 21, 21, 21, 96, 96, 128, 128],
        [9, 9, 25, 25, 37, 37, 37, 128, 128, 256, 256, 256, 256],
        [13, 13, 37, 37, 49, 49, 61, 61, 69, 192, 192, 352, 352, 416, 416, 480, 480],
        [21, 21, 45, 45, 69, 69, 89, 89, 97, 97, 97, 256, 256]
        + [416, 416, 608, 608, 704, 704, 704, 704],
    ]

    for r in range(1, 6):
        w = 2 * r + 1
        cart_resl = 1.0
        xb = XBin(cart_resl, 9e9)
        gr = xb.grid6
        pm = PHMap_u8f8()
        cen = xident_f4
        kcen = xb.key_of(xident_f4)
        bcen = xb.bincen_of(kcen)
        assert np.allclose(cen, bcen, atol=1e-4)
        phm = PHMap_u8f8()
        phm[xb.key_of(bcen)] = 1.0
        smeared = smear(xb, phm, radius=r, extrahalf=1, oddlast3=1, sphere=1)
        smeared2 = smear(xb, phm, radius=r, extrahalf=1, oddlast3=1, sphere=0)
        print(len(smeared) / len(smeared2))
        continue
        assert isinstance(smeared, PHMap_u8f8)
        assert len(smeared) == [83, 529, 1459, 3269, 6115][r - 1]

        k, v = smeared.items_array()
        x = xb.bincen_of(k)
        cart_dis = np.linalg.norm(bcen[0, :3, 3] - x[:, :3, 3], axis=1)
        d = 0.57787751
        uvals = np.arange(-2 * r, 2 * r + 0.001) * d
        assert np.allclose(np.unique(x[:, 0, 3]), uvals, atol=1e-4)
        assert np.allclose(np.unique(x[:, 1, 3]), uvals, atol=1e-4)
        assert np.allclose(np.unique(x[:, 2, 3]), uvals, atol=1e-4)
        assert sorted(Counter(x[:, 0, 3]).values()) == counts[r - 1]
        assert sorted(Counter(x[:, 1, 3]).values()) == counts[r - 1]
        assert sorted(Counter(x[:, 2, 3]).values()) == counts[r - 1]
        ori_dist = angle_of_3x3(x[:, :3, :3])
        assert np.allclose(np.unique(ori_dist), [0.0, 1.24466863])


def check_scores(s0, s1):
    not0 = np.sum(np.logical_or(s1 > 0, s0 > 0))
    frac_s1_gt_s0 = np.sum(s1 > s0) / not0
    frac_s1_ge_s0 = np.sum(s1 >= s0) / not0
    print(
        "score",
        "Ns0",
        np.sum(s0 > 0),
        "Ns1",
        np.sum(s1 > 0),
        "frac1>=0",
        frac_s1_ge_s0,
        "frac1>0",
        frac_s1_gt_s0,
    )
    return frac_s1_ge_s0, frac_s1_gt_s0, not0


def test_smear_bounding():
    N1 = 5_000
    N2 = 50_000
    cart_sd = 2
    xorig = hm.rand_xform(N1, cart_sd=cart_sd).astype("f4")
    sorig = np.exp(np.random.rand(N1))
    cart_resl = 1.0
    ori_resl = 20
    xb0 = XBin(cart_resl, ori_resl)
    xb2 = XBin(cart_resl * 2, ori_resl * 1.5)

    pm0 = PHMap_u8f8()
    pm0[xb0.key_of(xorig)] = sorig

    t = perf_counter()
    pm1 = smear(xb0, pm0, radius=1)
    t = perf_counter() - t
    print(
        f"fexpand {len(pm1) / len(pm0):7.2f} cell rate {int(len(pm1) / t):,} expand_rate {int(len(pm0) / t):,}"
    )

    x = hm.rand_xform(N2, cart_sd=cart_sd).astype("f4")
    s0 = pm0[xb0.key_of(x)]
    s1 = pm1[xb0.key_of(x)]
    ge, gt, not0 = check_scores(s0, s1)
    assert 0 == np.sum(np.logical_and(s0 > 0, s1 == 0))
    assert np.sum((s0 > 0) * (s1 == 0)) == 0
    assert ge > 0.99
    assert gt > 0.98

    pm20 = PHMap_u8f8()
    pm20[xb2.key_of(xorig)] = sorig
    t = perf_counter()
    pm2 = smear(xb2, pm20, radius=1)
    t = perf_counter() - t
    print(
        f"fexpand {len(pm2) / len(pm20):7.2f} cell rate {int(len(pm2) / t):,} expand_rate {int(len(pm20) / t):,}"
    )
    s2 = pm2[xb2.key_of(x)]
    ge, gt, not0 = check_scores(s0, s2)
    assert ge > 0.99
    assert gt > 0.99
    assert np.sum(np.logical_and(s0 > 0, s2 == 0)) / not0 < 0.001


def smear_bench():
    N = 1_000_000
    cart_sd = 5
    xorig = hm.rand_xform(N, cart_sd=cart_sd)
    sorig = np.exp(np.random.rand(N))
    cart_resl = 1.0
    ori_resl = 20
    xb0 = XBin(cart_resl, ori_resl)

    pm0 = PHMap_u8f8()
    pm0[xb0.key_of(xorig)] = sorig

    for rad in range(1, 2):
        t = perf_counter()
        pm1 = smear(xb0, pm0, radius=rad)
        t = perf_counter() - t
        print(
            f"rad {rad} relsize: {len(pm1) / len(pm0):7.2f} cell rate {int(len(pm1) / t):,} expand_rate {int(len   (pm0) / t):,}"
        )


if __name__ == "__main__":
    test_smear()
    test_smear_oddori()
    test_smear_oddori_sphere()
    test_smear_exhalf_oddori_sphere()
    test_smear_bounding()
