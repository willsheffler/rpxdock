from time import perf_counter
import numpy as np
from cppimport import import_hook
from sicdock.xbin import xbin_test, xbin

import homog as hm


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

    print(
        f"performance keys: {int(N/t):,} kfloat: {int(N/tf):,} cens: {int(N/tc):,}",
        f"cover {N/uniq} tgen: {int(N/tgen):,}",
    )


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


if __name__ == "__main__":
    assert xbin_test.TEST_XformHash_XformHash_bt24_BCC6()
    # test_key_of()
    # test_xbin_covrad()
