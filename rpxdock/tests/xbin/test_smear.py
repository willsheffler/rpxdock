from time import perf_counter
from collections import Counter
import numpy as np
import rpxdock.homog as hm
from rpxdock.geom import BCC6
from rpxdock.xbin import Xbin
from rpxdock.xbin.smear import smear
from rpxdock.phmap import PHMap_u8f8
from rpxdock.util import hackplot as plot

xident_f4 = np.eye(4).astype("f4")

def test_smear_one():
   for r in range(1, 6):
      w = 2 * r + 1
      cart_resl = 1.0
      xb = Xbin(cart_resl, 9e9)
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
      assert len(smeared) == w**3 + (w - 1)**3
      k, v = smeared.items_array()
      x = xb.bincen_of(k)
      cart_dis = np.linalg.norm(bcen[0, :3, 3] - x[:, :3, 3], axis=1)
      assert np.min(cart_dis) == 0
      # print(sorted(Counter(x[:, 0, 3]).values()))
      counts = [(w - 1)**2] * (w - 1) + [w**2] * w
      assert sorted(Counter(x[:, 0, 3]).values()) == counts
      assert sorted(Counter(x[:, 1, 3]).values()) == counts
      assert sorted(Counter(x[:, 2, 3]).values()) == counts
      ori_dist = hm.angle_of_3x3(x[:, :3, :3])
      assert np.allclose(np.unique(ori_dist), [0.0, 1.24466863])

def test_smear_one_oddori():
   for r in range(1, 6):
      w = 2 * r + 1
      cart_resl = 1.0
      xb = Xbin(cart_resl, 9e9)
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
      assert len(smeared) == w**3 + 8 * (w - 1)**3
      k, v = smeared.items_array()
      x = xb.bincen_of(k)
      cart_dis = np.linalg.norm(bcen[0, :3, 3] - x[:, :3, 3], axis=1)
      d = 0.57787751
      uvals = np.arange(-2 * r, 2 * r + 0.001) * d
      assert np.allclose(np.unique(x[:, 0, 3]), uvals, atol=1e-4)
      assert np.allclose(np.unique(x[:, 1, 3]), uvals, atol=1e-4)
      assert np.allclose(np.unique(x[:, 2, 3]), uvals, atol=1e-4)
      counts = [w**2] * w + [8 * (w - 1)**2] * (w - 1)
      assert sorted(Counter(x[:, 0, 3]).values()) == counts
      assert sorted(Counter(x[:, 1, 3]).values()) == counts
      assert sorted(Counter(x[:, 2, 3]).values()) == counts
      ori_dist = hm.angle_of_3x3(x[:, :3, :3])
      assert np.allclose(np.unique(ori_dist), [0.0, 1.24466863])

def test_smear_one_oddori_sphere():
   counts = [
      [5, 5, 9, 32, 32],
      [9, 9, 21, 21, 21, 96, 96, 128, 128],
      [9, 9, 25, 25, 37, 37, 37, 128, 128, 256, 256, 256, 256],
      [13, 13, 37, 37, 49, 49, 61, 61, 69, 192, 192, 352, 352, 416, 416, 480, 480],
      [21, 21, 45, 45, 69, 69, 89, 89, 97, 97, 97, 256, 256] +
      [416, 416, 608, 608, 704, 704, 704, 704],
   ]

   for r in range(1, 6):
      w = 2 * r + 1
      cart_resl = 1.0
      xb = Xbin(cart_resl, 9e9)
      gr = xb.grid6
      pm = PHMap_u8f8()
      cen = xident_f4
      kcen = xb.key_of(xident_f4)
      bcen = xb.bincen_of(kcen)
      assert np.allclose(cen, bcen, atol=1e-4)
      phm = PHMap_u8f8()
      phm[xb.key_of(bcen)] = 1.0
      smeared = smear(xb, phm, radius=r, extrahalf=0, oddlast3=1, sphere=1)
      smeared2 = smear(xb, phm, radius=r, extrahalf=0, oddlast3=1, sphere=1)
      print("smear sph/cube", len(smeared) / len(smeared2))
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
      ori_dist = hm.angle_of_3x3(x[:, :3, :3])
      assert np.allclose(np.unique(ori_dist), [0.0, 1.24466863])

def test_smear_one_exhalf_oddori_sphere():
   counts = [
      [5, 5, 9, 32, 32],
      [9, 9, 21, 21, 21, 96, 96, 128, 128],
      [9, 9, 25, 25, 37, 37, 37, 128, 128, 256, 256, 256, 256],
      [13, 13, 37, 37, 49, 49, 61, 61, 69, 192, 192, 352, 352, 416, 416, 480, 480],
      [21, 21, 45, 45, 69, 69, 89, 89, 97, 97, 97, 256, 256] +
      [416, 416, 608, 608, 704, 704, 704, 704],
   ]

   for r in range(1, 6):
      w = 2 * r + 1
      cart_resl = 1.0
      xb = Xbin(cart_resl, 9e9)
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
      print("smear exhalf sph/cube", len(smeared) / len(smeared2))
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
      ori_dist = hm.angle_of_3x3(x[:, :3, :3])
      assert np.allclose(np.unique(ori_dist), [0.0, 1.24466863])

def test_smear_two():
   for samp in range(10):
      for r in range(1, 5):
         w = 2 * r + 1
         cart_resl = 1.0
         ori_resl = 10
         xb = Xbin(cart_resl, ori_resl)
         gr = xb.grid6
         phm, phm0, phm1 = PHMap_u8f8(), PHMap_u8f8(), PHMap_u8f8()

         p = np.stack([np.eye(4), np.eye(4)]).astype("f4")
         p[:, :3, 3] = np.random.randn(2, 3) * (r / 2)
         p[1, :3, :3] = hm.rot(np.random.randn(3), ori_resl / 2, degrees=True)

         k = xb.key_of(p)
         phm[k] = np.array([1, 1], dtype="f8")
         smeared = smear(xb, phm, radius=r, extrahalf=1, oddlast3=1, sphere=1)
         allk, allv = smeared.items_array()

         smeared0 = [smear(xb, phm0, radius=r, extrahalf=1, oddlast3=1, sphere=1)]
         phm0[k[0]] = 1.0
         smeared0 = smear(xb, phm0, radius=r, extrahalf=1, oddlast3=1, sphere=1)
         allv0 = smeared0[allk]

         phm1[k[1]] = 1.0
         smeared1 = smear(xb, phm1, radius=r, extrahalf=1, oddlast3=1, sphere=1)
         allv1 = smeared1[allk]

         assert np.all(allv0 <= allv)
         assert np.all(allv1 <= allv)
         assert np.all(allv == np.maximum(allv0, allv1))

         d = np.linalg.norm(p[0, :3, 3] - p[1, :3, 3])
         s, s0, s1 = set(allk), set(smeared0.keys()), set(smeared1.keys())
         assert s0.issubset(s)
         assert s1.issubset(s)
         # print(len(s0.intersection(s1)) / len(s))

def test_smear_multiple():
   cart_resl = 1.0
   ori_resl = 10
   xb = Xbin(cart_resl, ori_resl)
   gr = xb.grid6

   for rad in range(1, 6):
      maxpts = [0, 20, 10, 6, 4, 2][rad]
      for npts in range(2, maxpts):
         w = 2 * rad + 1
         phm = PHMap_u8f8()
         phm0 = [PHMap_u8f8() for i in range(npts)]
         p = hm.hrot(np.random.randn(npts, 3), 1.5 * ori_resl, degrees=1, dtype="f4")
         p[:, :3, 3] = np.random.randn(npts, 3) * 1.5 * rad
         k = xb.key_of(p)
         phm[k] = np.ones(npts)
         smeared = smear(xb, phm, radius=rad, extrahalf=1, oddlast3=1, sphere=1)
         allk, allv = smeared.items_array()
         sallk = set(allk)
         allv0 = np.empty((npts, len(allk)))
         sets = list()
         for i in range(npts):
            phm0[i][k[i]] = 1.0
            smr = smear(xb, phm0[i], radius=rad, extrahalf=1, oddlast3=1, sphere=1)
            allv0[i] = smr[allk]
            assert np.all(allv0[i] <= allv)
            s0 = set(smr.keys())
            assert s0.issubset(sallk)
            sets.append(s0)
         # nisect = np.mean([len(a.intersection(b)) for a in sets for b in sets])
         # print(rad, npts, nisect / len(sets[0]))
         # assert np.all(allv == np.max(allv0, axis=0))

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

def test_smear_one_bounding():
   N1 = 5_000
   N2 = 50_000
   cart_sd = 2
   xorig = hm.rand_xform(N1, cart_sd=cart_sd).astype("f4")
   sorig = np.exp(np.random.rand(N1))
   cart_resl = 1.0
   ori_resl = 20
   xb0 = Xbin(cart_resl, ori_resl)
   xb2 = Xbin(cart_resl * 2, ori_resl * 1.5)

   pm0 = PHMap_u8f8()
   pm0[xb0.key_of(xorig)] = sorig

   t = perf_counter()
   pm1 = smear(xb0, pm0, radius=1)
   t = perf_counter() - t
   print(
      f"fexpand {len(pm1) / len(pm0):7.2f}",
      f"cell rate {int(len(pm1) / t):,}",
      f" expand_rate {int(len(pm0) / t):,}",
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
   xb0 = Xbin(cart_resl, ori_resl)

   pm0 = PHMap_u8f8()
   pm0[xb0.key_of(xorig)] = sorig

   for rad in range(1, 2):
      t = perf_counter()
      pm1 = smear(xb0, pm0, radius=rad)
      t = perf_counter() - t
      print(
         f"rad {rad} relsize: {len(pm1) / len(pm0):7.2f} ",
         f"cell rate {int(len(pm1) / t):,}",
         f"expand_rate {int(len   (pm0) / t):,}",
      )

def test_smear_one_kernel():
   spherefudge = {
      (1, 0): 0.3734 + 0.0001,
      (1, 1): 0.0361 + 0.0001,
      (2, 0): 0.0474 + 0.0001,
      (2, 1): 0.2781 + 0.0001,
      (3, 0): 0.0148 + 0.0001,
      (3, 1): 0.1347 + 0.0001,
      (4, 0): 0.0510 + 0.0001,
      (4, 1): 0.1583 + 0.0001,
   }
   cone4Dfudge = {
      (1, 0): 0.0091 + 0.0001,
      (1, 1): 0.1417 + 0.0001,
      (2, 0): 0.1163 + 0.0001,
      (2, 1): 0.1221 + 0.0001,
      (3, 0): 0.1208 + 0.0001,
      (3, 1): 0.1304 + 0.0001,
      (4, 0): 0.1213 + 0.0001,
      (4, 1): 0.1240 + 0.0001,
   }
   parab4fudge = {
      (1, 0): 0.0041 + 0.0001,
      (1, 1): 0.1688 + 0.0001,
      (2, 0): 0.1347 + 0.0001,
      (2, 1): 0.1436 + 0.0001,
      (3, 0): 0.1402 + 0.0001,
      (3, 1): 0.1532 + 0.0001,
      (4, 0): 0.1413 + 0.0001,
      (4, 1): 0.1448 + 0.0001,
   }

   N = 8
   # plot.subplots(4, N // 2, rowmajor=True)
   for rad in range(N):
      exhalf = (rad) % 2
      rad = (rad) // 2 + 1
      # print("rad", rad, "exhalf", exhalf)
      w = 2 * rad + 1
      cart_resl = 1.0
      xb = Xbin(cart_resl, 9e9)
      gr = xb.grid6
      pm = PHMap_u8f8()
      cen = xident_f4
      kcen = xb.key_of(xident_f4)
      bcen = xb.bincen_of(kcen)
      assert np.allclose(cen, bcen, atol=1e-4)
      phm = PHMap_u8f8()
      phm[xb.key_of(bcen)] = 1.0
      grid_r2 = xb.grid6.neighbor_sphere_radius_square_cut(rad, exhalf)
      d2 = np.arange(grid_r2 + 1)

      # kern = np.exp(-d2 / grid_r2 * 2)

      kern0 = 1 - (d2 / grid_r2)**(1 / 2)  # 1/R
      kern1 = 1 - (d2 / grid_r2)**(2 / 2)  # 1/R**2
      kern2 = 1 - (d2 / grid_r2)**(3 / 2)  # 1/R**3 uniform in R
      kern3 = np.ones(len(d2))

      # plot.scatter(np.sqrt(d2), kern1, show=0)
      # smeared = smear(xb, phm, rad, exhalf, oddlast3=1, sphere=1, kernel=kern0)
      # k, v = smeared.items_array()

      vals = []
      for ikrn in [0, 1, 2, 3]:
         kern = vars()["kern%i" % ikrn]
         smeared = smear(xb, phm, rad, exhalf, oddlast3=1, sphere=1, kernel=kern)
         k, v = smeared.items_array()
         vals.append(v)
         # plot.hist(v, title="kern%i" % ikrn, show=0)
         # print(rad, "kern%i" % ikrn, np.sum(v))
      assert np.all(vals[0] <= vals[1])
      assert np.all(vals[1] <= vals[2])
      assert np.all(vals[2] <= vals[3])

      vol0 = np.sum(vals[0])
      vol1 = np.sum(vals[1])
      vol2 = np.sum(vals[2])
      vol3 = np.sum(vals[3])
      spherevol = 4 / 3 * np.pi * grid_r2**(3 / 2)
      parab4vol = 1 / 6 * np.pi**2 * grid_r2**(3 / 2)  # ?? close enough...
      cone4Dvol = spherevol / 4

      assert np.abs(1 - vol0 / cone4Dvol) < cone4Dfudge[rad, exhalf]
      assert np.abs(1 - vol1 / parab4vol) < parab4fudge[rad, exhalf]
      assert np.abs(1 - vol3 / spherevol) < spherefudge[rad, exhalf]

      # print(np.sum(vals[3]) / spherevol)
   # plot.show()

if __name__ == "__main__":
   # test_smear_one()
   # test_smear_one_oddori()
   # test_smear_one_oddori_sphere()
   # test_smear_one_exhalf_oddori_sphere()
   # test_smear_one_bounding()
   test_smear_two()
   # test_smear_multiple()
   # test_smear_one_kernel()
