import numpy as np
import rpxdock.homog as hm
import rpxdock

def rand_xform_sphere(n, radius, maxang=0):
   t = radius * 2 * (np.random.rand(3 * n + 10, 3) - 0.5)
   t = t[np.linalg.norm(t, axis=-1) <= radius]
   assert len(t) >= n
   ang = (np.random.rand(n) - 0.5) * 2 * maxang
   axs = (np.random.randn(n, 3), )
   x = hm.hrot(axs, ang, degrees=True)
   x[:, :3, 3] = t[:n]
   return x

def test_rpxhier(hscore, N=1000):
   # mean error rates are lower than what you might infer from thresholds below

   print("size base", len(hscore.base.score_map))
   for i in range(len(hscore.hier)):
      print("size   ", i, len(hscore.hier[i]))

   scorebase = hscore.base.score_map
   xforms = scorebase.xforms(N)
   scores = scorebase[xforms]
   for i, hier in enumerate(hscore.hier):
      scores_above = hscore.hier[i][xforms]
      olap = np.sum(scores_above != 0) / len(xforms)
      bfrac = np.sum(scores_above >= scores) / len(xforms)
      print(
         f"{i} baseolap {olap:5.3f} boundfrac {bfrac:5.3f}",
         f"cart {hier.xbin.cart_resl:7.3f} ori {hier.xbin.ori_resl:7.3f}",
      )
      assert olap > 0.75
      assert bfrac > 0.7

   avg_base_ptrb_olap, avg_base_ptrb_bfrac = list(), list()
   for i, hier in enumerate(hscore.hier):
      cart_extent = hier.attr.cart_extent
      ori_extent = hier.attr.ori_extent
      perturb = rand_xform_sphere(len(xforms), cart_extent / 2, ori_extent / 2)
      xperturb = xforms @ perturb.astype("f")
      scores_above = hscore.hier[i][xperturb]
      ptrb_olap = np.sum(scores_above != 0) / len(xperturb)
      ptrb_bfrac = np.sum(scores_above >= scores) / len(xperturb)
      print(
         f"ptrb {i} baseolap {ptrb_olap:5.3f} boundfrac {ptrb_bfrac:5.3f}",
         f"cart {cart_extent:7.3f} ori {ori_extent:7.3f}",
      )
      avg_base_ptrb_olap.append(ptrb_olap)
      avg_base_ptrb_bfrac.append(ptrb_bfrac)
      assert ptrb_olap > 0.65
      assert ptrb_bfrac > 0.65
   assert np.mean(avg_base_ptrb_olap) > 0.77
   assert np.mean(avg_base_ptrb_bfrac) > 0.78

   avg_recov = list()
   for i, hier in enumerate(hscore.hier):
      xforms = hier.xforms(N)
      scores = hier.phmap.items_array(len(xforms))[1]
      scoresx = hier[xforms]
      recov = np.sum(scoresx > 0) / np.sum(scores > 0)
      print(i, "self recov", np.sum(scoresx > 0) / np.sum(scores > 0))
      assert recov > 0.95

   avg_ptrb_olap, avg_ptrb_bfrac = list(), list()
   for i, hier in enumerate(hscore.hier):
      xforms = hier.xforms(N)
      scoresx = hier[xforms]
      cart_extent0 = hier.attr.cart_extent
      ori_extent0 = hier.attr.ori_extent
      for j in range(i):
         cart_extent = hscore.hier[j].attr.cart_extent - cart_extent0
         ori_extent = hscore.hier[j].attr.ori_extent - ori_extent0
         perturb = rand_xform_sphere(len(xforms), cart_extent / 2, ori_extent / 2)
         xperturb = xforms @ perturb.astype("f")
         scores_above = hscore.hier[j][xperturb]
         olap = np.sum(scores_above > 0) / np.sum(scoresx > 0)
         bfrac = np.sum(scores_above >= scoresx) / len(xforms)
         print(
            f"{i} {j} olap {olap:5.3f} bfrac {bfrac:5.3f}",
            f"{cart_extent:7.3f} {ori_extent:7.3f}",
         )
         avg_ptrb_olap.append(olap)
         avg_ptrb_bfrac.append(bfrac)
         assert olap > 0.6
         assert bfrac > 0.58
   assert np.mean(avg_ptrb_olap) > 0.70
   assert np.mean(avg_ptrb_bfrac) > 0.70

   print("bounding metrics", np.mean(avg_base_ptrb_bfrac), np.mean(avg_ptrb_bfrac))

   # diff = np.diff(scores, axis=0)
   # print(diff.shape)
   # for i in range(len(hscore.hier) - 1):
   # print(i, np.sum(diff[i] <= 0) / diff[i].size)

   # print("base", hscore.base.xbin.cart_resl, hscore.base.xbin.ori_resl)
   # for h in hscore.hier:
   #     print("hier", h.xbin.cart_resl, h.xbin.ori_resl)
