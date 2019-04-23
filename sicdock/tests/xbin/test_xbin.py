from time import perf_counter
import numpy as np
from cppimport import import_hook
from sicdock.xbin import xbin_test, xbin

import homog as hm

"""
data from cpp test (commented out because pytest always prints it)
  bt24_BCC6 4.000/ 30.0 cr 3.537 dt 0.875 da  0.754 x2k: 115.394ns k2x: 108.414ns 46531.591    1521
  bt24_BCC6 2.000/ 20.0 cr 1.857 dt 0.914 da  0.887 x2k: 133.190ns k2x: 116.280ns 14615.713    2855
  bt24_BCC6 1.000/ 15.0 cr 0.907 dt 0.904 da  0.771 x2k: 115.988ns k2x: 104.792ns  4417.715    7917
  bt24_BCC6 0.500/ 10.0 cr 0.456 dt 0.905 da  0.850 x2k: 120.220ns k2x: 129.110ns  1195.048   16693
  bt24_BCC6 0.250/  5.0 cr 0.232 dt 0.897 da  0.870 x2k: 117.767ns k2x: 112.731ns  1115.061  112604
  bt24_BCC6 0.110/  3.3 cr 0.099 dt 0.863 da  0.838 x2k: 133.243ns k2x: 120.677ns   738.212 1048576

"""


def test_xbin_cpp():
    assert xbin_test.TEST_XformHash_XformHash_bt24_BCC6()


def test_create_binner():
    binner = xbin.XBin(1.0, 15.0, 256.0)
    binner2 = xbin.XBin_float(1.0, 15.0, 256.0)
    assert binner
    assert binner2


def test_key_of():
    N = 1_000_00
    binparam = (0.3, 5.0, 256.0)

    tgen = perf_counter()
    x = hm.rand_xform(N)
    xfloat = hm.rand_xform(N).astype("f4")
    tgen = perf_counter() - tgen

    t = perf_counter()
    k = xbin.key_of(x, *binparam)
    t = perf_counter() - t

    tf = perf_counter()

    k = xbin.key_of(xfloat, *binparam)
    tf = perf_counter() - tf

    tc = perf_counter()
    c = xbin.bincen_of(k, *binparam)
    tc = perf_counter() - tc

    uniq = len(np.unique(k))

    xbin.key_of(x[:10], 1, 20)
    xbin.key_of(x[:10], 1)
    xbin.key_of(x[:10])

    print(
        f"performance keys: {int(N/t):,} kfloat: {int(N/tf):,} cens: {int(N/tc):,}",
        f"cover {N/uniq} tgen: {int(N/tgen):,}",
    )


def test_key_of_pairs():
    N = 10_000
    N1, N2 = 100, 1000
    x1 = hm.rand_xform(N1)
    x2 = hm.rand_xform(N2)
    i1 = np.random.randint(0, N1, N)
    i2 = np.random.randint(0, N2, N)
    p = np.stack([i1, i2], axis=1)
    k1 = xbin.key_of_pairs(p, x1, x2)

    px1 = x1[p[:, 0]]
    px2 = x2[p[:, 1]]
    k2 = xbin.key_of(np.linalg.inv(px1) @ px2)
    assert np.all(k1 == k2)

    k3 = xbin.key_of_pairs2(i1, i2, x1, x2)
    assert np.all(k1 == k3)


def test_xbin_covrad():
    niter = 30
    nsamp = 1000
    for i in range(niter):
        cart_resl = np.random.rand() * 10 + 0.1
        ori_resl = np.random.rand() * 50 + 2.5
        xforms = hm.rand_xform(nsamp)
        xb = xbin.XBin(cart_resl, ori_resl, 512)
        idx = xb.key_of(xforms)
        cen = xb.bincen_of(idx)
        cart_dist = np.linalg.norm(xforms[..., :3, 3] - cen[..., :3, 3], axis=-1)
        ori_dist = hm.angle_of(hm.hinv(cen) @ xforms)
        # if not np.all(cart_dist < cart_resl):
        # print('ori_resl', ori_resl, 'nside:', xb.ori_nside,
        # 'max(cart_dist):', np.max(cart_dist), cart_resl)
        # if not np.all(cart_dist < cart_resl):
        # print('ori_resl', ori_resl, 'nside:', xb.ori_nside,
        # 'max(ori_dist):', np.max(ori_dist))
        assert np.all(cart_dist < cart_resl)
        assert np.all(ori_dist < ori_resl / 180 * np.pi)


def test_key_of_pairs2_ss():
    N = 10_000
    N1, N2 = 100, 1000
    x1 = hm.rand_xform(N1)
    x2 = hm.rand_xform(N2)
    ss1 = np.random.randint(0, 3, N1)
    ss2 = np.random.randint(0, 3, N2)
    i1 = np.random.randint(0, N1, N)
    i2 = np.random.randint(0, N2, N)
    # p = np.stack([i1, i2], axis=1)
    k1 = xbin.key_of_pairs2(i1, i2, x1, x2)

    kss = xbin.key_of_pairs2_ss(i1, i2, ss1, ss2, x1, x2)
    assert np.all(k1 == np.right_shift(np.left_shift(kss, 4), 4))
    assert np.all(ss1[i1] == np.right_shift(np.left_shift(kss, 0), 62))
    assert np.all(ss2[i2] == np.right_shift(np.left_shift(kss, 2), 62))


if __name__ == "__main__":
    # assert xbin_test.TEST_XformHash_XformHash_bt24_BCC6()
    # test_key_of()
    # test_xbin_covrad()
    test_key_of_pairs2_ss()
