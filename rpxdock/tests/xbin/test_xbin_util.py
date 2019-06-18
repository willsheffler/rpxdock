import _pickle
from time import perf_counter
import numpy as np
from rpxdock.xbin import Xbin_double, Xbin_float
import rpxdock.xbin.xbin_util as xu
from rpxdock.homog import angle_of_3x3
from rpxdock.geom import bcc
from rpxdock import phmap

import rpxdock.homog as hm

def test_key_of_pairs():
   xb = Xbin_double(1, 20)
   N = 10000
   N1, N2 = 100, 1000
   x1 = hm.rand_xform(N1)
   x2 = hm.rand_xform(N2)
   i1 = np.random.randint(0, N1, N)
   i2 = np.random.randint(0, N2, N)
   p = np.stack([i1, i2], axis=1)
   k1 = xu.key_of_pairs(xb, p, x1, x2)

   px1 = x1[p[:, 0]]
   px2 = x2[p[:, 1]]
   k2 = xb.key_of(np.linalg.inv(px1) @ px2)
   assert np.all(k1 == k2)

   k3 = xu.key_of_selected_pairs(xb, i1, i2, x1, x2)
   assert np.all(k1 == k3)

def test_sskey_of_selected_pairs():
   xb = Xbin_float()
   N = 10000
   N1, N2 = 100, 1000
   x1 = hm.rand_xform(N1).astype("f4")
   x2 = hm.rand_xform(N2).astype("f4")
   ss1 = np.random.randint(0, 3, N1)
   ss2 = np.random.randint(0, 3, N2)
   i1 = np.random.randint(0, N1, N)
   i2 = np.random.randint(0, N2, N)
   # p = np.stack([i1, i2], axis=1)
   k1 = xu.key_of_selected_pairs(xb, i1, i2, x1, x2)

   kss = xu.sskey_of_selected_pairs(xb, i1, i2, ss1, ss2, x1, x2)
   assert np.all(k1 == np.right_shift(np.left_shift(kss, 4), 4))
   assert np.all(ss1[i1] == np.right_shift(np.left_shift(kss, 0), 62))
   assert np.all(ss2[i2] == np.right_shift(np.left_shift(kss, 2), 62))

   idx = np.stack([i1, i2], axis=1)
   kss2 = xu.sskey_of_selected_pairs(xb, idx, ss1, ss2, x1, x2)
   assert np.all(kss == kss2)

def test_ssmap_of_selected_pairs():
   phm = phmap.PHMap_u8f8()
   xb = Xbin_float()
   N = 10000
   N1, N2 = 100, 1000
   x1 = hm.rand_xform(N1).astype("f4")
   x2 = hm.rand_xform(N2).astype("f4")
   ss1 = np.random.randint(0, 3, N1)
   ss2 = np.random.randint(0, 3, N2)
   i1 = np.random.randint(0, N1, N)
   i2 = np.random.randint(0, N2, N)
   idx = np.stack([i1, i2], axis=1)

   keys = xu.sskey_of_selected_pairs(xb, idx, ss1, ss2, x1, x2)

   vals = xu.ssmap_of_selected_pairs(xb, phm, idx, ss1, ss2, x1, x2)
   assert np.all(vals == 0)

   phm.default = 7
   vals = xu.ssmap_of_selected_pairs(xb, phm, idx, ss1, ss2, x1, x2)
   assert np.all(vals == 7)
   phm.default = 0

   vals0 = np.random.rand(len(keys))
   phm[keys] = vals0
   vals = xu.ssmap_of_selected_pairs(xb, phm, idx, ss1, ss2, x1, x2)
   assert np.all(vals == phm[keys])

def test_map_of_selected_pairs():
   phm = phmap.PHMap_u8f8()
   xb = Xbin_float()
   N = 10000
   N1, N2 = 100, 1000
   x1 = hm.rand_xform(N1).astype("f4")
   x2 = hm.rand_xform(N2).astype("f4")
   i1 = np.random.randint(0, N1, N)
   i2 = np.random.randint(0, N2, N)
   idx = np.stack([i1, i2], axis=1)

   vals = xu.map_of_selected_pairs(xb, phm, idx, x1, x2)
   assert np.all(vals == 0)

   phm.default = 7
   vals = xu.map_of_selected_pairs(xb, phm, idx, x1, x2)
   assert np.all(vals == 7)
   phm.default = 0

   keys = xu.key_of_selected_pairs(xb, idx, x1, x2)
   vals0 = np.random.rand(len(keys))
   phm[keys] = vals0
   vals = xu.map_of_selected_pairs(xb, phm, idx, x1, x2)
   assert np.all(vals == phm[keys])

def test_selected_pairs_pos():
   xb = Xbin_float()
   N = 1
   N1, N2 = 100, 1000
   x1 = hm.rand_xform(N1).astype("f4")
   x2 = hm.rand_xform(N2).astype("f4")
   p1, p2 = hm.rand_xform(), hm.rand_xform()
   i1 = np.random.randint(0, N1, N)
   i2 = np.random.randint(0, N2, N)
   idx = np.stack([i1, i2], axis=1)

   k1 = xu.key_of_selected_pairs(xb, i1, i2, p1 @ x1, p2 @ x2)
   k2 = xu.key_of_selected_pairs(xb, i1, i2, x1, x2, p1, p2)
   assert np.all(k1 == k2)

if __name__ == "__main__":
   test_key_of_pairs()
   test_sskey_of_selected_pairs()
   test_ssmap_of_selected_pairs()
   test_map_of_selected_pairs()
   test_selected_pairs_pos()
