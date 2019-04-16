from time import perf_counter
import numpy as np
from cppimport import import_hook
from tcdock.xbin import xbin_test, xbin

import homog as hm

def test_create_binner():
    binner = xbin._XBin_double(1.0, 15.0, 256.0)
    print(binner)

def test_get_keys():
    N = 1_000

    binner = xbin._XBin_double(0.3, 5.0, 256.0)
    tgen = perf_counter()
    x = hm.rand_xform(N)
    tgen = perf_counter()-tgen

    t = perf_counter()
    k = xbin.get_keys_double(binner, x)
    t = perf_counter()-t

    tc = perf_counter()
    c = xbin.get_centers_double(binner, k)
    tc = perf_counter()-tc

    uniq = len(np.unique(k))

    print(f"performance keys {int(N/t):,} cens {int(N/tc):,} cover {N/uniq} tgen {int(N/tgen):,}")



def test_xbin_covrad():
    niter = 30
    nsamp = 1000
    for i in range(niter):
        cart_resl = np.random.rand() * 10 + 0.1
        ori_resl = np.random.rand() * 50 + 2.5
        xforms = hm.rand_xform(nsamp)
        xb = xbin._XBin_double(cart_resl, ori_resl, 512)
        idx = xbin.get_keys_double(xb, xforms)
        cen = xbin.get_centers_double(xb, idx)
        cart_dist = np.linalg.norm(
            xforms[..., :3, 3] - cen[..., :3, 3], axis=-1)
        ori_dist = hm.angle_of(hm.hinv(cen) @ xforms)
        # if not np.all(cart_dist < cart_resl):
        # print('ori_resl', ori_resl, 'nside:', xb.ori_nside,
        # 'max(cart_dist):', np.max(cart_dist), cart_resl)
        # if not np.all(cart_dist < cart_resl):
        # print('ori_resl', ori_resl, 'nside:', xb.ori_nside,
        # 'max(ori_dist):', np.max(ori_dist))
        assert np.all(cart_dist < cart_resl)
        assert np.all(ori_dist < ori_resl / 180 * np.pi)


if __name__ == '__main__':
    xbin_test.TEST_XformHash_XformHash_bt24_BCC6()
    test_get_keys()
    test_xbin_covrad()
