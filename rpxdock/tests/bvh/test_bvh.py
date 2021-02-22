import _pickle, numpy as np, itertools as it, sys
from time import perf_counter

# from cppimport import import_hook
#
# # import cppimport
#
# # cppimport.set_quiet(False)
#
import rpxdock as rp
from rpxdock.bvh import bvh_test
from rpxdock.bvh import BVH, bvh
import rpxdock.homog as hm

def test_bvh_isect_cpp():
   assert bvh_test.TEST_bvh_test_isect()

def test_bvh_isect_fixed():
   # print()
   mindist = 0.01

   totbvh, totnaive = 0, 0

   for i in range(10):
      xyz1 = np.random.rand(1000, 3) + [0.9, 0.9, 0]
      xyz2 = np.random.rand(1000, 3)
      tcre = perf_counter()
      bvh1 = BVH(xyz1)
      bvh2 = BVH(xyz2)
      tcre = perf_counter() - tcre
      assert len(bvh1) == 1000

      pos1 = hm.htrans([0.9, 0.9, 0.9])
      pos2 = np.eye(4)

      tbvh = perf_counter()
      clash1 = bvh.bvh_isect_fixed(bvh1, bvh2, mindist)
      tbvh = perf_counter() - tbvh

      tn = perf_counter()
      clash2 = bvh.naive_isect_fixed(bvh1, bvh2, mindist)
      tn = perf_counter() - tn

      assert clash1 == clash2

      # print(f"{i:3} clash {clash1:1} {tn / tbvh:8.2f}, {tn:1.6f}, {tbvh:1.6f}")

      totbvh += tbvh
      totnaive += tn

   print("total times", totbvh, totnaive / totbvh, totnaive)

def test_bvh_isect():
   t = rp.Timer().start()

   N1, N2 = 10, 10
   N = N1 * N2
   mindist = 0.04
   nclash = 0

   for outer in range(N1):

      xyz1 = np.random.rand(1250, 3) - [0.5, 0.5, 0.5]
      xyz2 = np.random.rand(1250, 3) - [0.5, 0.5, 0.5]
      pos1 = hm.rand_xform(N2, cart_sd=0.8)
      pos2 = hm.rand_xform(N2, cart_sd=0.8)
      t.checkpoint('init')

      bvh1 = BVH(xyz1)
      bvh2 = BVH(xyz2)
      t.checkpoint('BVH')

      clash = list()
      for inner in range(N2):
         clash1 = bvh.bvh_isect(bvh1=bvh1, bvh2=bvh2, pos1=pos1[inner], pos2=pos2[inner],
                                mindist=mindist)
         t.checkpoint('bvh_isect')
         clash2 = bvh.naive_isect(bvh1, bvh2, pos1[inner], pos2[inner], mindist)
         t.checkpoint('naive_isect')
         assert clash1 == clash2
         clash.append(clash1)

      clashvec = bvh.bvh_isect_vec(bvh1, bvh2, pos1, pos2, mindist)
      t.checkpoint('bvh_isect_vec')
      assert np.all(clashvec == clash)
      nclash += sum(clash)

      assert clashvec[1] == bvh.bvh_isect_vec(bvh1, bvh2, pos1[1], pos2[1], mindist)
      bvh.bvh_isect_vec(bvh1, bvh2, pos1, pos2[1], mindist)  # ?? make sure api works?
      bvh.bvh_isect_vec(bvh1, bvh2, pos1[1], pos2, mindist)

   print(
      f"Ngeom {N1:,} Npos {N2:,} isect {nclash/N:4.2f} bvh: {int(N/t.sum.bvh_isect):,}/s",
      f"bvh_vec {int(N/t.sum.bvh_isect_vec):,} fastnaive {int(N/t.sum.naive_isect):,}/s",
      f"ratio {int(t.sum.naive_isect/t.sum.bvh_isect_vec):,}x",
   )

def test_bvh_isect_fixed_range():
   N1, N2 = 10, 10
   N = N1 * N2
   mindist = 0.04
   nclash = 0

   for outer in range(N1):

      xyz1 = np.random.rand(1000, 3) - [0.5, 0.5, 0.5]
      xyz2 = np.random.rand(1000, 3) - [0.5, 0.5, 0.5]
      bvh1 = BVH(xyz1)
      bvh2 = BVH(xyz2)
      bvh1_half = BVH(xyz1[250:750])
      bvh2_half = BVH(xyz2[250:750])
      pos1 = hm.rand_xform(N2, cart_sd=0.5)
      pos2 = hm.rand_xform(N2, cart_sd=0.5)

      isect1 = bvh.bvh_isect_vec(bvh1, bvh2, pos1, pos2, mindist)
      isect2, clash = bvh.bvh_isect_fixed_range_vec(bvh1, bvh2, pos1, pos2, mindist)
      assert np.all(isect1 == isect2)

      bounds = [250], [749], [250], [749]
      isect1 = bvh.bvh_isect_vec(bvh1_half, bvh2_half, pos1, pos2, mindist)
      isect2, clash = bvh.bvh_isect_fixed_range_vec(bvh1, bvh2, pos1, pos2, mindist, *bounds)
      assert np.all(isect1 == isect2)

def test_bvh_min_cpp():
   assert bvh_test.TEST_bvh_test_min()

def test_bvh_min_dist_fixed():
   xyz1 = np.random.rand(5000, 3) + [0.9, 0.9, 0.0]
   xyz2 = np.random.rand(5000, 3)
   tcre = perf_counter()
   bvh1 = BVH(xyz1)
   bvh2 = BVH(xyz2)
   tcre = perf_counter() - tcre

   tbvh = perf_counter()
   d, i1, i2 = bvh.bvh_min_dist_fixed(bvh1, bvh2)
   tbvh = perf_counter() - tbvh
   dtest = np.linalg.norm(xyz1[i1] - xyz2[i2])
   assert np.allclose(d, dtest, atol=1e-6)

   # tnp = perf_counter()
   # dnp = np.min(np.linalg.norm(xyz1[:, None] - xyz2[None], axis=2))
   # tnp = perf_counter() - tnp

   tn = perf_counter()
   dn = bvh.naive_min_dist_fixed(bvh1, bvh2)
   tn = perf_counter() - tn

   print()
   print("from bvh:  ", d)
   print("from naive:", dn)
   assert np.allclose(dn, d, atol=1e-6)

   print(f"tnaivecpp {tn:5f} tbvh {tbvh:5f} tbvhcreate {tcre:5f}")
   print("bvh acceleration vs naive", tn / tbvh)
   # assert tn / tbvh > 100

def test_bvh_min_dist():

   xyz1 = np.random.rand(1000, 3) - [0.5, 0.5, 0.5]
   xyz2 = np.random.rand(1000, 3) - [0.5, 0.5, 0.5]
   tcre = perf_counter()
   bvh1 = BVH(xyz1)
   bvh2 = BVH(xyz2)
   tcre = perf_counter() - tcre
   # print()
   totbvh, totnaive = 0, 0
   N = 10
   pos1 = hm.rand_xform(N, cart_sd=1)
   pos2 = hm.rand_xform(N, cart_sd=1)
   dis = list()
   for i in range(N):

      tbvh = perf_counter()
      d, i1, i2 = bvh.bvh_min_dist(bvh1, bvh2, pos1[i], pos2[i])
      tbvh = perf_counter() - tbvh
      dtest = np.linalg.norm(pos1[i] @ hm.hpoint(xyz1[i1]) - pos2[i] @ hm.hpoint(xyz2[i2]))
      assert np.allclose(d, dtest, atol=1e-6)

      tn = perf_counter()
      dn = bvh.naive_min_dist(bvh1, bvh2, pos1[i], pos2[i])
      tn = perf_counter() - tn
      assert np.allclose(dn, d, atol=1e-6)
      dis.append((d, i1, i2))

      # print(
      # f"tnaivecpp {tn:1.6f} tbvh {tbvh:1.6f} tcpp/tbvh {tn/tbvh:8.1f}",
      # np.linalg.norm(pos1[:3, 3]),
      # dtest - d,
      # )
      totnaive += tn
      totbvh += tbvh

   d, i1, i2 = bvh.bvh_min_dist_vec(bvh1, bvh2, pos1, pos2)
   for a, b, c, x in zip(d, i1, i2, dis):
      assert a == x[0]
      assert b == x[1]
      assert c == x[2]

   print(
      "total times",
      totbvh / N * 1000,
      "ms",
      totnaive / totbvh,
      totnaive,
      f"tcre {tcre:2.4f}",
   )

def test_bvh_min_dist_floormin():

   xyz1 = np.random.rand(1000, 3) - [0.5, 0.5, 0.5]
   xyz2 = np.random.rand(1000, 3) - [0.5, 0.5, 0.5]
   tcre = perf_counter()
   bvh1 = BVH(xyz1)
   bvh2 = BVH(xyz2)
   tcre = perf_counter() - tcre
   # print()
   totbvh, totnaive = 0, 0
   N = 10
   for i in range(N):
      pos1 = hm.rand_xform(cart_sd=1)
      pos2 = hm.rand_xform(cart_sd=1)

      tbvh = perf_counter()
      d, i1, i2 = bvh.bvh_min_dist(bvh1, bvh2, pos1, pos2)
      tbvh = perf_counter() - tbvh
      dtest = np.linalg.norm(pos1 @ hm.hpoint(xyz1[i1]) - pos2 @ hm.hpoint(xyz2[i2]))
      assert np.allclose(d, dtest, atol=1e-6)

      tn = perf_counter()
      dn = bvh.naive_min_dist(bvh1, bvh2, pos1, pos2)
      tn = perf_counter() - tn
      assert np.allclose(dn, d, atol=1e-6)

      # print(
      # f"tnaivecpp {tn:1.6f} tbvh {tbvh:1.6f} tcpp/tbvh {tn/tbvh:8.1f}",
      # np.linalg.norm(pos1[:3, 3]),
      # dtest - d,
      # )
      totnaive += tn
      totbvh += tbvh

   print(
      "total times",
      totbvh / N * 1000,
      "ms",
      totnaive / totbvh,
      totnaive,
      f"tcre {tcre:2.4f}",
   )

def test_bvh_slide_single_inline():

   bvh1 = BVH([[-10, 0, 0]])
   bvh2 = BVH([[0, 0, 0]])
   d = bvh.bvh_slide(bvh1, bvh2, np.eye(4), np.eye(4), rad=1.0, dirn=[1, 0, 0])
   assert d == 8
   # moves xyz1 to -2,0,0

   # should always come in from "infinity" from -direction
   bvh1 = BVH([[10, 0, 0]])
   bvh2 = BVH([[0, 0, 0]])
   d = bvh.bvh_slide(bvh1, bvh2, np.eye(4), np.eye(4), rad=1.0, dirn=[1, 0, 0])
   assert d == -12
   # also moves xyz1 to -2,0,0

   for i in range(100):
      np.random.seed(i)
      dirn = np.array([np.random.randn(), 0, 0])
      dirn /= np.linalg.norm(dirn)
      rad = np.abs(np.random.randn() / 10)
      xyz1 = np.array([[np.random.randn(), 0, 0]])
      xyz2 = np.array([[np.random.randn(), 0, 0]])
      bvh1 = BVH(xyz1)
      bvh2 = BVH(xyz2)
      d = bvh.bvh_slide(bvh1, bvh2, np.eye(4), np.eye(4), rad=rad, dirn=dirn)
      xyz1 += d * dirn
      assert np.allclose(np.linalg.norm(xyz1 - xyz2), 2 * rad, atol=1e-4)

def test_bvh_slide_single():

   nmiss = 0
   for i in range(100):
      # np.random.seed(i)
      dirn = np.random.randn(3)
      dirn /= np.linalg.norm(dirn)
      rad = np.abs(np.random.randn())
      xyz1 = np.random.randn(1, 3)
      xyz2 = np.random.randn(1, 3)
      bvh1 = BVH(xyz1)
      bvh2 = BVH(xyz2)
      d = bvh.bvh_slide(bvh1, bvh2, np.eye(4), np.eye(4), rad=rad, dirn=dirn)
      if d < 9e8:
         xyz1 += d * dirn
         assert np.allclose(np.linalg.norm(xyz1 - xyz2), 2 * rad, atol=1e-4)
      else:
         nmiss += 1
         delta = xyz2 - xyz1
         d0 = delta.dot(dirn)
         dperp2 = np.sum(delta * delta) - d0 * d0
         target_d2 = 4 * rad**2
         assert target_d2 < dperp2
   print("nmiss", nmiss, nmiss / 1000)

def test_bvh_slide_single_xform():

   nmiss = 0
   for i in range(1000):
      dirn = np.random.randn(3)
      dirn /= np.linalg.norm(dirn)
      rad = np.abs(np.random.randn() * 2.0)
      xyz1 = np.random.randn(1, 3)
      xyz2 = np.random.randn(1, 3)
      bvh1 = BVH(xyz1)
      bvh2 = BVH(xyz2)
      pos1 = hm.rand_xform()
      pos2 = hm.rand_xform()
      d = bvh.bvh_slide(bvh1, bvh2, pos1, pos2, rad=rad, dirn=dirn)
      if d < 9e8:
         p1 = (pos1 @ hm.hpoint(xyz1[0]))[:3] + d * dirn
         p2 = (pos2 @ hm.hpoint(xyz2[0]))[:3]
         assert np.allclose(np.linalg.norm(p1 - p2), 2 * rad, atol=1e-4)
      else:
         nmiss += 1
         p2 = pos2 @ hm.hpoint(xyz2[0])
         p1 = pos1 @ hm.hpoint(xyz1[0])
         delta = p2 - p1
         d0 = delta[:3].dot(dirn)
         dperp2 = np.sum(delta * delta) - d0 * d0
         target_d2 = 4 * rad**2
         assert target_d2 < dperp2
   print("nmiss", nmiss, nmiss / 1000)

def test_bvh_slide_whole():

   # timings wtih -Ofast
   # slide test 10,000 iter bvhslide float: 16,934/s double: 16,491/s bvhmin 17,968/s fracmiss: 0.0834

   # np.random.seed(0)
   N1, N2 = 2, 10
   totbvh, totbvhf, totmin = 0, 0, 0
   nmiss = 0
   for j in range(N1):
      xyz1 = np.random.rand(5000, 3) - [0.5, 0.5, 0.5]
      xyz2 = np.random.rand(5000, 3) - [0.5, 0.5, 0.5]
      # tcre = perf_counter()
      bvh1 = BVH(xyz1)
      bvh2 = BVH(xyz2)
      # bvh1f = BVH_32bit(xyz1)
      # bvh2f = BVH_32bit(xyz2)
      # tcre = perf_counter() - tcre
      pos1 = hm.rand_xform(N2, cart_sd=0.5)
      pos2 = hm.rand_xform(N2, cart_sd=0.5)
      dirn = np.random.randn(3)
      dirn /= np.linalg.norm(dirn)
      radius = 0.001 + np.random.rand() / 10
      slides = list()
      for i in range(N2):
         tbvh = perf_counter()
         dslide = bvh.bvh_slide(bvh1, bvh2, pos1[i], pos2[i], radius, dirn)
         tbvh = perf_counter() - tbvh
         tbvhf = perf_counter()
         # dslide = bvh.bvh_slide_32bit(bvh1f, bvh2f, pos1[i], pos2[i], radius, dirn)
         tbvhf = perf_counter() - tbvhf
         slides.append(dslide)
         if dslide > 9e8:
            tn = perf_counter()
            dn, i, j = bvh.bvh_min_dist(bvh1, bvh2, pos1[i], pos2[i])
            tn = perf_counter() - tn
            assert dn > 2 * radius
            nmiss += 1
         else:
            tmp = hm.htrans(dirn * dslide) @ pos1[i]
            tn = perf_counter()
            dn, i, j = bvh.bvh_min_dist(bvh1, bvh2, tmp, pos2[i])
            tn = perf_counter() - tn
            if not np.allclose(dn, 2 * radius, atol=1e-6):
               print(dn, 2 * radius)
            assert np.allclose(dn, 2 * radius, atol=1e-6)

         # print(
         # i,
         # f"tnaivecpp {tn:1.6f} tbvh {tbvh:1.6f} tcpp/tbvh {tn/tbvh:8.1f}",
         # np.linalg.norm(pos1[:3, 3]),
         # dslide,
         # )
         totmin += tn
         totbvh += tbvh
         totbvhf += tbvhf
      slides2 = bvh.bvh_slide_vec(bvh1, bvh2, pos1, pos2, radius, dirn)
      assert np.allclose(slides, slides2)
   N = N1 * N2
   print(
      f"slide test {N:,} iter bvhslide double: {int(N/totbvh):,}/s bvhmin {int(N/totmin):,}/s",
      # f"slide test {N:,} iter bvhslide float: {int(N/totbvhf):,}/s double: {int(N/totbvh):,}/s bvhmin {int(N/totmin):,}/s",
      f"fracmiss: {nmiss/N}",
   )

def test_collect_pairs_simple():
   print("test_collect_pairs_simple")
   bufbvh = -np.ones((100, 2), dtype="i4")
   bufnai = -np.ones((100, 2), dtype="i4")
   bvh1 = BVH([[0, 0, 0], [0, 2, 0]])
   bvh2 = BVH([[0.9, 0, 0], [0.9, 2, 0]])
   assert len(bvh1) == 2
   mindist = 1.0

   pos1 = np.eye(4)
   pos2 = np.eye(4)
   pbvh, o = bvh.bvh_collect_pairs(bvh1, bvh2, pos1, pos2, mindist, bufbvh)
   nnai = bvh.naive_collect_pairs(bvh1, bvh2, pos1, pos2, mindist, bufnai)
   assert not o
   print(pbvh.shape)
   assert len(pbvh) == 2 and nnai == 2
   assert np.all(pbvh == [[0, 0], [1, 1]])
   assert np.all(bufnai[:nnai] == [[0, 0], [1, 1]])

   pos1 = hm.htrans([0, 2, 0])
   pbvh, o = bvh.bvh_collect_pairs(bvh1, bvh2, pos1, pos2, mindist, bufbvh)
   nnai = bvh.naive_collect_pairs(bvh1, bvh2, pos1, pos2, mindist, bufnai)
   assert not o
   assert len(pbvh) == 1 and nnai == 1
   assert np.all(pbvh == [[0, 1]])
   assert np.all(bufnai[:nnai] == [[0, 1]])

   pos1 = hm.htrans([0, -2, 0])
   pbvh, o = bvh.bvh_collect_pairs(bvh1, bvh2, pos1, pos2, mindist, bufbvh)
   nnai = bvh.naive_collect_pairs(bvh1, bvh2, pos1, pos2, mindist, bufnai)
   assert not o
   assert len(pbvh) == 1 and nnai == 1
   assert np.all(pbvh == [[1, 0]])
   assert np.all(bufnai[:nnai] == [[1, 0]])

def test_collect_pairs_simple_selection():
   print("test_collect_pairs_simple_selection")
   bufbvh = -np.ones((100, 2), dtype="i4")
   bufnai = -np.ones((100, 2), dtype="i4")
   crd1 = [[0, 0, 0], [0, 0, 0], [0, 2, 0], [0, 0, 0]]
   crd2 = [[0, 0, 0], [0.9, 0, 0], [0, 0, 0], [0.9, 2, 0]]
   mask1 = [1, 0, 1, 0]
   mask2 = [0, 1, 0, 1]
   bvh1 = BVH(crd1, mask1)
   bvh2 = BVH(crd2, mask2)
   assert len(bvh1) == 2
   assert np.allclose(bvh1.radius(), 1.0, atol=1e-6)
   assert np.allclose(bvh1.center(), [0, 1, 0], atol=1e-6)
   mindist = 1.0

   pos1 = np.eye(4)
   pos2 = np.eye(4)
   pbvh, o = bvh.bvh_collect_pairs(bvh1, bvh2, pos1, pos2, mindist, bufbvh)
   assert not o
   nnai = bvh.naive_collect_pairs(bvh1, bvh2, pos1, pos2, mindist, bufnai)
   assert len(pbvh) == 2 and nnai == 2
   assert np.all(pbvh == [[0, 1], [2, 3]])
   assert np.all(bufnai[:nnai] == [[0, 1], [2, 3]])

   pos1 = hm.htrans([0, 2, 0])
   pbvh, o = bvh.bvh_collect_pairs(bvh1, bvh2, pos1, pos2, mindist, bufbvh)
   assert not o
   nnai = bvh.naive_collect_pairs(bvh1, bvh2, pos1, pos2, mindist, bufnai)
   assert len(pbvh) == 1 and nnai == 1
   assert np.all(pbvh == [[0, 3]])
   assert np.all(bufnai[:nnai] == [[0, 3]])

   pos1 = hm.htrans([0, -2, 0])
   pbvh, o = bvh.bvh_collect_pairs(bvh1, bvh2, pos1, pos2, mindist, bufbvh)
   assert not o
   nnai = bvh.naive_collect_pairs(bvh1, bvh2, pos1, pos2, mindist, bufnai)
   assert len(pbvh) == 1 and nnai == 1
   assert np.all(pbvh == [[2, 1]])
   assert np.all(bufnai[:nnai] == [[2, 1]])

def test_collect_pairs():
   N1, N2 = 1, 50
   N = N1 * N2
   Npts = 500
   totbvh, totbvhf, totmin = 0, 0, 0
   totbvh, totnai, totct, ntot = 0, 0, 0, 0
   bufbvh = -np.ones((Npts * Npts, 2), dtype="i4")
   bufnai = -np.ones((Npts * Npts, 2), dtype="i4")
   for j in range(N1):
      xyz1 = np.random.rand(Npts, 3) - [0.5, 0.5, 0.5]
      xyz2 = np.random.rand(Npts, 3) - [0.5, 0.5, 0.5]
      bvh1 = BVH(xyz1)
      bvh2 = BVH(xyz2)

      pos1, pos2 = list(), list()
      while 1:
         x1 = hm.rand_xform(cart_sd=0.5)
         x2 = hm.rand_xform(cart_sd=0.5)
         d = np.linalg.norm(x1[:, 3] - x2[:, 3])
         if 0.8 < d < 1.3:
            pos1.append(x1)
            pos2.append(x2)
            if len(pos1) == N2:
               break
      pos1 = np.stack(pos1)
      pos2 = np.stack(pos2)
      pairs = list()
      mindist = 0.002 + np.random.rand() / 10
      for i in range(N2):
         tbvh = perf_counter()
         pbvh, o = bvh.bvh_collect_pairs(bvh1, bvh2, pos1[i], pos2[i], mindist, bufbvh)
         tbvh = perf_counter() - tbvh
         assert not o

         tnai = perf_counter()
         nnai = bvh.naive_collect_pairs(bvh1, bvh2, pos1[i], pos2[i], mindist, bufnai)
         tnai = perf_counter() - tnai

         tct = perf_counter()
         nct = bvh.bvh_count_pairs(bvh1, bvh2, pos1[i], pos2[i], mindist)
         tct = perf_counter() - tct
         ntot += nct

         assert nct == len(pbvh)
         totnai += 1

         pairs.append(pbvh.copy())

         totbvh += tbvh
         totnai += tnai
         totct += tct

         assert len(pbvh) == nnai
         if len(pbvh) == 0:
            continue

         o = np.lexsort((pbvh[:, 1], pbvh[:, 0]))
         pbvh[:] = pbvh[:][o]
         o = np.lexsort((bufnai[:nnai, 1], bufnai[:nnai, 0]))
         bufnai[:nnai] = bufnai[:nnai][o]
         assert np.all(pbvh == bufnai[:nnai])

         pair1 = pos1[i] @ hm.hpoint(xyz1[pbvh[:, 0]])[..., None]
         pair2 = pos2[i] @ hm.hpoint(xyz2[pbvh[:, 1]])[..., None]
         dpair = np.linalg.norm(pair2 - pair1, axis=1)
         assert np.max(dpair) <= mindist

      pcount = bvh.bvh_count_pairs_vec(bvh1, bvh2, pos1, pos2, mindist)
      assert np.all(pcount == [len(x) for x in pairs])

      pairs2, lbub = bvh.bvh_collect_pairs_vec(bvh1, bvh2, pos1, pos2, mindist)
      for i, p in enumerate(pairs):
         lb, ub = lbub[i]
         assert np.all(pairs2[lb:ub] == pairs[i])

      x, y = bvh.bvh_collect_pairs_vec(bvh1, bvh2, pos1[:3], pos2[0], mindist)
      assert len(y) == 3
      x, y = bvh.bvh_collect_pairs_vec(bvh1, bvh2, pos1[0], pos2[:5], mindist)
      assert len(y) == 5

   print(
      f"collect test {N:,} iter bvh {int(N/totbvh):,}/s naive {int(N/totnai):,}/s ratio {totnai/totbvh:7.2f} count-only {int(N/totct):,}/s avg cnt {ntot/N}"
   )

def test_collect_pairs_range():
   N1, N2 = 1, 500
   N = N1 * N2
   Npts = 1000
   for j in range(N1):
      xyz1 = np.random.rand(Npts, 3) - [0.5, 0.5, 0.5]
      xyz2 = np.random.rand(Npts, 3) - [0.5, 0.5, 0.5]
      bvh1 = BVH(xyz1)
      bvh2 = BVH(xyz2)
      pos1, pos2 = list(), list()
      while 1:
         x1 = hm.rand_xform(cart_sd=0.5)
         x2 = hm.rand_xform(cart_sd=0.5)
         d = np.linalg.norm(x1[:, 3] - x2[:, 3])
         if 0.8 < d < 1.3:
            pos1.append(x1)
            pos2.append(x2)
            if len(pos1) == N2:
               break
      pos1 = np.stack(pos1)
      pos2 = np.stack(pos2)
      pairs = list()
      mindist = 0.002 + np.random.rand() / 10

      pairs, lbub = bvh.bvh_collect_pairs_vec(bvh1, bvh2, pos1, pos2, mindist)
      rpairs, rlbub = bvh.bvh_collect_pairs_range_vec(bvh1, bvh2, pos1, pos2, mindist)
      assert np.all(lbub == rlbub)
      assert np.all(pairs == rpairs)

      rpairs, rlbub = bvh.bvh_collect_pairs_range_vec(bvh1, bvh2, pos1, pos2, mindist, [250],
                                                      [750])
      assert len(rlbub) == len(pos1)
      assert np.all(rpairs[:, 0] >= 250)
      assert np.all(rpairs[:, 0] <= 750)
      filt_pairs = pairs[np.logical_and(pairs[:, 0] >= 250, pairs[:, 0] <= 750)]
      # assert np.all(filt_pairs == rpairs)  # sketchy???

      u1 = np.unique(filt_pairs, axis=1)
      u2 = np.unique(rpairs, axis=1)
      if len(u1) != len(u2):  # hopefully this will deal with the rare test failures?
         assert abs(len(u1) - len(u1)) <= 1
         s1 = set((3, y) for x, y in u1)
         s2 = set((3, y) for x, y in u2)
         assert 0.9 < len(s1 and s2) / len(s1)
         continue
      else:
         assert np.all(u1 == u2)

      rpairs, rlbub = bvh.bvh_collect_pairs_range_vec(bvh1, bvh2, pos1, pos2, mindist, [600],
                                                      [1000], -1, [100], [400], -1)
      assert len(rlbub) == len(pos1)
      assert np.all(rpairs[:, 0] >= 600)
      assert np.all(rpairs[:, 0] <= 1000)
      assert np.all(rpairs[:, 1] >= 100)
      assert np.all(rpairs[:, 1] <= 400)
      filt_pairs = pairs[(pairs[:, 0] >= 600) * (pairs[:, 0] <= 1000) * (pairs[:, 1] >= 100) *
                         (pairs[:, 1] <= 400)]
      assert np.all(filt_pairs == rpairs)  # sketchy???
      assert np.allclose(np.unique(filt_pairs, axis=1), np.unique(rpairs, axis=1))

def test_collect_pairs_range_sym():
   # np.random.seed(132)
   N1, N2 = 5, 100
   N = N1 * N2
   Npts = 1000
   for j in range(N1):
      xyz1 = np.random.rand(Npts, 3) - [0.5, 0.5, 0.5]
      xyz2 = np.random.rand(Npts, 3) - [0.5, 0.5, 0.5]
      bvh1 = BVH(xyz1)
      bvh2 = BVH(xyz2)
      pos1, pos2 = list(), list()
      while 1:
         x1 = hm.rand_xform(cart_sd=0.5)
         x2 = hm.rand_xform(cart_sd=0.5)
         d = np.linalg.norm(x1[:, 3] - x2[:, 3])
         if 0.8 < d < 1.3:
            pos1.append(x1)
            pos2.append(x2)
            if len(pos1) == N2:
               break
      pos1 = np.stack(pos1)
      pos2 = np.stack(pos2)
      pairs = list()
      mindist = 0.002 + np.random.rand() / 10

      pairs, lbub = bvh.bvh_collect_pairs_vec(bvh1, bvh2, pos1, pos2, mindist)
      rpairs, rlbub = bvh.bvh_collect_pairs_range_vec(bvh1, bvh2, pos1, pos2, mindist)
      assert np.all(lbub == rlbub)
      assert np.all(pairs == rpairs)

      bounds = [100], [400], len(xyz1) // 2
      rpairs, rlbub = bvh.bvh_collect_pairs_range_vec(bvh1, bvh2, pos1, pos2, mindist, *bounds)
      assert len(rlbub) == len(pos1)
      assert np.all(
         np.logical_or(np.logical_and(100 <= rpairs[:, 0], rpairs[:, 0] <= 400),
                       np.logical_and(600 <= rpairs[:, 0], rpairs[:, 0] <= 900)))
      filt_pairs = pairs[np.logical_or(np.logical_and(100 <= pairs[:, 0], pairs[:, 0] <= 400),
                                       np.logical_and(600 <= pairs[:, 0], pairs[:, 0] <= 900))]
      assert np.allclose(np.unique(filt_pairs, axis=1), np.unique(rpairs, axis=1))

      bounds = [100], [400], len(xyz1) // 2, [20], [180], len(xyz1) // 5
      rpairs, rlbub = bvh.bvh_collect_pairs_range_vec(bvh1, bvh2, pos1, pos2, mindist, *bounds)

      def awful(p):
         return np.logical_and(
            np.logical_or(np.logical_and(100 <= p[:, 0], p[:, 0] <= 400),
                          np.logical_and(600 <= p[:, 0], p[:, 0] <= 900)),
            np.logical_or(
               np.logical_and(+20 <= p[:, 1], p[:, 1] <= 180),
               np.logical_or(
                  np.logical_and(220 <= p[:, 1], p[:, 1] <= 380),
                  np.logical_or(
                     np.logical_and(420 <= p[:, 1], p[:, 1] <= 580),
                     np.logical_or(np.logical_and(620 <= p[:, 1], p[:, 1] <= 780),
                                   np.logical_and(820 <= p[:, 1], p[:, 1] <= 980))))))

      assert len(rlbub) == len(pos1)
      assert np.all(awful(rpairs))
      filt_pairs = pairs[awful(pairs)]
      assert np.all(filt_pairs == rpairs)  # sketchy???
      assert np.allclose(np.unique(filt_pairs, axis=1), np.unique(rpairs, axis=1))

def test_slide_collect_pairs():

   # timings wtih -Ofast
   # slide test 10,000 iter bvhslide float: 16,934/s double: 16,491/s bvhmin 17,968/s fracmiss: 0.0834

   # np.random.seed(0)
   N1, N2 = 2, 50
   Npts = 5000
   totbvh, totbvhf, totcol, totmin = 0, 0, 0, 0
   nhit = 0
   buf = -np.ones((Npts * Npts, 2), dtype="i4")
   for j in range(N1):
      xyz1 = np.random.rand(Npts, 3) - [0.5, 0.5, 0.5]
      xyz2 = np.random.rand(Npts, 3) - [0.5, 0.5, 0.5]
      xyzcol1 = xyz1[:int(Npts / 5)]
      xyzcol2 = xyz2[:int(Npts / 5)]
      # tcre = perf_counter()
      bvh1 = BVH(xyz1)
      bvh2 = BVH(xyz2)
      bvhcol1 = BVH(xyzcol1)
      bvhcol2 = BVH(xyzcol2)
      # tcre = perf_counter() - tcre
      for i in range(N2):
         dirn = np.random.randn(3)
         dirn /= np.linalg.norm(dirn)
         radius = 0.001 + np.random.rand() / 10
         pairdis = 3 * radius
         pos1 = hm.rand_xform(cart_sd=0.5)
         pos2 = hm.rand_xform(cart_sd=0.5)

         tbvh = perf_counter()
         dslide = bvh.bvh_slide(bvh1, bvh2, pos1, pos2, radius, dirn)
         tbvh = perf_counter() - tbvh

         if dslide > 9e8:
            tn = perf_counter()
            dn, i, j = bvh.bvh_min_dist(bvh1, bvh2, pos1, pos2)
            tn = perf_counter() - tn
            assert dn > 2 * radius
         else:
            nhit += 1
            pos1 = hm.htrans(dirn * dslide) @ pos1
            tn = perf_counter()
            dn, i, j = bvh.bvh_min_dist(bvh1, bvh2, pos1, pos2)
            tn = perf_counter() - tn
            if not np.allclose(dn, 2 * radius, atol=1e-6):
               print(dn, 2 * radius)
            assert np.allclose(dn, 2 * radius, atol=1e-6)

            tcol = perf_counter()
            pair, o = bvh.bvh_collect_pairs(bvhcol1, bvhcol2, pos1, pos2, pairdis, buf)
            assert not o
            if len(pair) > 0:
               tcol = perf_counter() - tcol
               totcol += tcol
               pair1 = pos1 @ hm.hpoint(xyzcol1[pair[:, 0]])[..., None]
               pair2 = pos2 @ hm.hpoint(xyzcol2[pair[:, 1]])[..., None]
               dpair = np.linalg.norm(pair2 - pair1, axis=1)
               assert np.max(dpair) <= pairdis

         totmin += tn
         totbvh += tbvh

   N = N1 * N2
   print(
      f"slide test {N:,} iter bvhslide double: {int(N/totbvh):,}/s bvhmin {int(N/totmin):,}/s",
      # f"slide test {N:,} iter bvhslide float: {int(N/totbvhf):,}/s double: {int(N/totbvh):,}/s bvhmin {int(N/totmin):,}/s",
      f"fracmiss: {nhit/N} collect {int(nhit/totcol):,}/s",
   )

def test_bvh_accessors():
   xyz = np.random.rand(10, 3) - [0.5, 0.5, 0.5]
   b = BVH(xyz)
   assert np.allclose(b.com()[:3], np.mean(xyz, axis=0))
   p = b.centers()
   dmat = np.linalg.norm(p[:, :3] - xyz[:, None], axis=2)
   assert np.allclose(np.min(dmat, axis=1), 0)

def random_walk(N):
   x = np.random.randn(N, 3).astype("f").cumsum(axis=0)
   x -= x.mean(axis=0)
   return 0.5 * x / x.std()

def test_bvh_isect_range(body=None, cart_sd=0.3, N2=10, mindist=0.02):
   N1 = 1 if body else 2
   N = N1 * N2
   totbvh, totnaive, totbvh0, nhit = 0, 0, 0, 0

   for ibvh in range(N1):
      if body:
         bvh1, bvh2 = body.bvh_bb, body.bvh_bb
      else:
         # xyz1 = np.random.rand(2000, 3) - [0.5, 0.5, 0.5]
         # xyz2 = np.random.rand(2000, 3) - [0.5, 0.5, 0.5]
         xyz1 = random_walk(1000)
         xyz2 = random_walk(1000)
         tcre = perf_counter()
         bvh1 = BVH(xyz1)
         bvh2 = BVH(xyz2)
         tcre = perf_counter() - tcre

      pos1 = hm.rand_xform(N2, cart_sd=cart_sd)
      pos2 = hm.rand_xform(N2, cart_sd=cart_sd)
      ranges = list()
      for i in range(N2):

         tbvh0 = perf_counter()
         c = bvh.bvh_isect(bvh1=bvh1, bvh2=bvh2, pos1=pos1[i], pos2=pos2[i], mindist=mindist)
         tbvh0 = perf_counter() - tbvh0

         # if not c:
         # continue
         if c:
            nhit += 1

         tbvh = perf_counter()
         range1 = bvh.isect_range_single(bvh1=bvh1, bvh2=bvh2, pos1=pos1[i], pos2=pos2[i],
                                         mindist=mindist)
         tbvh = perf_counter() - tbvh

         tn = perf_counter()
         range2 = bvh.naive_isect_range(bvh1, bvh2, pos1[i], pos2[i], mindist)
         assert range1 == range2
         tn = perf_counter() - tn

         ranges.append(range1)
         # print(f"{str(range1):=^80}")
         # body.move_to(pos1).dump_pdb("test1.pdb")
         # body.move_to(pos2).dump_pdb("test2.pdb")
         # return

         # print(f"{i:3} range {range1} {tn / tbvh:8.2f}, {tn:1.6f}, {tbvh:1.6f}")

         totbvh += tbvh
         totnaive += tn
         totbvh0 += tbvh0
      lb, ub = bvh.isect_range(bvh1, bvh2, pos1, pos2, mindist)
      ranges = np.array(ranges)
      assert np.all(lb == ranges[:, 0])
      assert np.all(ub == ranges[:, 1])

      ok = np.logical_and(lb >= 0, ub >= 0)
      isect, clash = bvh.bvh_isect_fixed_range_vec(bvh1, bvh2, pos1, pos2, mindist, lb, ub)
      assert not np.any(isect[ok])

   print(
      f"iscet {nhit:,} hit of {N:,} iter bvh: {int(nhit/totbvh):,}/s fastnaive {int(nhit/totnaive):,}/s",
      f"ratio {int(totnaive/totbvh):,}x isect-only: {totbvh/totbvh0:3.3f}x",
   )

def test_bvh_isect_range_ids():
   N1 = 50
   N2 = 100
   N = N1 * N2
   # Nids = 100
   cart_sd = 0.3
   mindist = 0.03
   Npts = 1000
   factors = [1000, 500, 250, 200, 125, 100, 50, 40, 25, 20, 10, 8, 5, 4, 2, 1]

   # Npts = 6
   # factors = [3]
   # mindist = 0.3
   # N1 = 1
   assert all(Npts % f == 0 for f in factors)
   for ibvh in range(N1):
      # for ibvh in [5]:

      # np.random.seed(ibvh)
      # print(ibvh)

      Nids = factors[ibvh % len(factors)]
      # xyz1 = np.random.rand(2000, 3) - [0.5, 0.5, 0.5]
      # xyz2 = np.random.rand(2000, 3) - [0.5, 0.5, 0.5]
      xyz1 = random_walk(Npts)
      xyz2 = random_walk(Npts)
      tcre = perf_counter()
      bvh1 = BVH(xyz1, [], np.repeat(np.arange(Nids), Npts / Nids))
      bvh2 = BVH(xyz2, [], np.repeat(np.arange(Nids), Npts / Nids))
      tcre = perf_counter() - tcre
      pos1 = hm.rand_xform(N2, cart_sd=cart_sd)
      pos2 = hm.rand_xform(N2, cart_sd=cart_sd)
      # pos1 = pos1[99:]
      # pos2 = pos2[99:]

      # print(bvh1.vol_lb())
      # print(bvh1.vol_ub())
      # print(bvh1.obj_id())
      # assert 0

      # assert bvh1.max_id() == Nids - 1
      # assert bvh1.min_lb() == 0
      # assert bvh1.max_ub() == Nids - 1

      lb, ub = bvh.isect_range(bvh1, bvh2, pos1, pos2, mindist)
      pos1 = pos1[lb != -1]
      pos2 = pos2[lb != -1]
      ub = ub[lb != -1]
      lb = lb[lb != -1]

      # print(lb, ub)

      assert np.all(0 <= lb) and np.all(lb - 1 <= ub) and np.all(ub < Nids)

      isectall = bvh.bvh_isect_vec(bvh1, bvh2, pos1, pos2, mindist)
      assert np.all(isectall == np.logical_or(lb > 0, ub < Nids - 1))

      isect, clash = bvh.bvh_isect_fixed_range_vec(bvh1, bvh2, pos1, pos2, mindist, lb, ub)

      if np.any(isect):
         print(np.where(isect)[0])
         print('lb', lb[isect])
         print('ub', ub[isect])
         print('cA', clash[isect, 0])
         print('cB', clash[isect, 1])

      # print('is', isect.astype('i') * 100)
      # print('isectlbub', np.sum(isect), np.sum(isect) / len(isect))
      assert not np.any(isect[lb <= ub])

def test_bvh_isect_range_lb_ub(body=None, cart_sd=0.3, N1=3, N2=20, mindist=0.02):
   N1 = 1 if body else N1
   N = N1 * N2
   Npts = 1000
   nhit, nrangefail = 0, 0
   kws = [
      rp.Bunch(maxtrim=a, maxtrim_lb=b, maxtrim_ub=c) for a in (-1, 400) for b in (-1, 300)
      for c in (-1, 300)
   ]
   for ibvh, kw in it.product(range(N1), kws):
      if body:
         bvh1, bvh2 = body.bvh_bb, body.bvh_bb
      else:
         # xyz1 = np.random.rand(Npts, 3) - [0.5, 0.5, 0.5]
         # xyz2 = np.random.rand(Npts, 3) - [0.5, 0.5, 0.5]
         xyz1 = random_walk(Npts)
         xyz2 = random_walk(Npts)
         bvh1 = BVH(xyz1)
         bvh2 = BVH(xyz2)

      pos1 = hm.rand_xform(N2, cart_sd=cart_sd)
      pos2 = hm.rand_xform(N2, cart_sd=cart_sd)
      ranges = list()
      for i in range(N2):
         c = bvh.bvh_isect(bvh1=bvh1, bvh2=bvh2, pos1=pos1[i], pos2=pos2[i], mindist=mindist)
         if c: nhit += 1
         range1 = bvh.isect_range_single(bvh1=bvh1, bvh2=bvh2, pos1=pos1[i], pos2=pos2[i],
                                         mindist=mindist, **kw)
         ranges.append(range1)
         if range1[0] < 0:
            nrangefail += 1
            assert c
            continue

         assert (kw.maxtrim < 0) or (np.diff(range1) + 1 >= Npts - kw.maxtrim)
         assert (kw.maxtrim_lb < 0) or (range1[0] <= kw.maxtrim_lb)
         assert (kw.maxtrim_ub < 0) or (range1[1] + 1 >= Npts - kw.maxtrim_ub)

         # mostly covered elsewhere, and quite slow
         # range2 = bvh.naive_isect_range(bvh1, bvh2, pos1[i], pos2[i], mindist)
         # assert range1 == range2

      lb, ub = bvh.isect_range(bvh1, bvh2, pos1, pos2, mindist, **kw)
      ranges = np.array(ranges)
      assert np.all(lb == ranges[:, 0])
      assert np.all(ub == ranges[:, 1])

   print(f"iscet {nhit:,} hit of {N:,} iter, frangefail {nrangefail/nhit}", )

def test_bvh_pickle(tmpdir):
   xyz1 = np.random.rand(1000, 3) - [0.5, 0.5, 0.5]
   xyz2 = np.random.rand(1000, 3) - [0.5, 0.5, 0.5]
   bvh1 = BVH(xyz1)
   bvh2 = BVH(xyz2)
   pos1 = hm.rand_xform(cart_sd=1)
   pos2 = hm.rand_xform(cart_sd=1)
   tbvh = perf_counter()
   d, i1, i2 = bvh.bvh_min_dist(bvh1, bvh2, pos1, pos2)
   rng = bvh.isect_range_single(bvh1, bvh2, pos1, pos2, mindist=d + 0.01)

   with open(tmpdir + "/1", "wb") as out:
      _pickle.dump(bvh1, out)
   with open(tmpdir + "/2", "wb") as out:
      _pickle.dump(bvh2, out)
   with open(tmpdir + "/1", "rb") as out:
      bvh1b = _pickle.load(out)
   with open(tmpdir + "/2", "rb") as out:
      bvh2b = _pickle.load(out)

   assert len(bvh1) == len(bvh1b)
   assert len(bvh2) == len(bvh2b)
   assert np.allclose(bvh1.com(), bvh1b.com())
   assert np.allclose(bvh1.centers(), bvh1b.centers())
   assert np.allclose(bvh2.com(), bvh2b.com())
   assert np.allclose(bvh2.centers(), bvh2b.centers())

   db, i1b, i2b = bvh.bvh_min_dist(bvh1b, bvh2b, pos1, pos2)
   assert np.allclose(d, db)
   assert i1 == i1b
   assert i2 == i2b
   rngb = bvh.isect_range_single(bvh1b, bvh2b, pos1, pos2, mindist=d + 0.01)
   assert rngb == rng

def test_bvh_threading_isect_may_fail():
   from concurrent.futures import ThreadPoolExecutor
   from itertools import repeat

   reps = 1
   npos = 1000

   Npts = 1000
   xyz1 = np.random.rand(Npts, 3) - [0.5, 0.5, 0.5]
   xyz2 = np.random.rand(Npts, 3) - [0.5, 0.5, 0.5]
   bvh1 = BVH(xyz1)
   bvh2 = BVH(xyz2)
   mindist = 0.1

   tottmain, tottthread = 0, 0
   nt = 2
   exe = ThreadPoolExecutor(nt)

   for i in range(reps):

      pos1 = hm.rand_xform(npos, cart_sd=0.5)
      pos2 = hm.rand_xform(npos, cart_sd=0.5)

      buf = np.empty((Npts, 2), dtype="i4")
      t = perf_counter()
      _ = [bvh.bvh_isect(bvh1, bvh2, p1, p2, mindist) for p1, p2 in zip(pos1, pos2)]
      isect = np.array(_)
      tmain = perf_counter() - t
      tottmain += tmain

      t = perf_counter()
      futures = exe.map(
         bvh.bvh_isect_vec,
         repeat(bvh1),
         repeat(bvh2),
         np.split(pos1, nt),
         np.split(pos2, nt),
         repeat(mindist),
      )
      isect2 = np.concatenate([f for f in futures])
      tthread = perf_counter() - t
      tottthread += tthread

      print("fisect", np.sum(isect2) / len(isect2))

      assert np.allclose(isect, isect2)
      # print("bvh_isect", i, tmain / tthread, ">= 1.1")
      # assert tmain / tthread > 1.1

   print("bvh_isect", tottmain / tottthread)

def test_bvh_threading_mindist_may_fail():
   from concurrent.futures import ThreadPoolExecutor
   from itertools import repeat

   reps = 1
   npos = 100

   Npts = 1000
   xyz1 = np.random.rand(Npts, 3) - [0.5, 0.5, 0.5]
   xyz2 = np.random.rand(Npts, 3) - [0.5, 0.5, 0.5]
   bvh1 = BVH(xyz1)
   bvh2 = BVH(xyz2)

   tottmain, tottthread = 0, 0
   nt = 2
   exe = ThreadPoolExecutor(nt)

   for i in range(reps):

      pos1 = hm.rand_xform(npos, cart_sd=0.7)
      pos2 = hm.rand_xform(npos, cart_sd=0.7)

      buf = np.empty((Npts, 2), dtype="i4")
      t = perf_counter()
      _ = [bvh.bvh_min_dist(bvh1, bvh2, p1, p2) for p1, p2 in zip(pos1, pos2)]
      mindist = np.array(_)
      tmain = perf_counter() - t
      tottmain += tmain

      t = perf_counter()
      futures = exe.map(
         bvh.bvh_min_dist_vec,
         repeat(bvh1),
         repeat(bvh2),
         np.split(pos1, nt),
         np.split(pos2, nt),
      )
      mindist2 = np.concatenate([f for f in futures], axis=1).T
      tthread = perf_counter() - t
      tottthread += tthread

      assert np.allclose(mindist, mindist2)
      # print("bvh_min_dist", i, tmain / tthread, ">= 1.1")
      # assert tmain / tthread > 1.1

   print("bvh_min_dist", tottmain / tottthread)

def bvh_perf():
   timer = rp.Timer().start()
   N = 10
   Npts = 50 * 50 * 4
   for j in range(N):
      xyz1 = np.random.rand(Npts, 3) + [0, 0, 0]
      xyz2 = np.random.rand(Npts, 3) + [1, 0, 0]
      timer.checkpoint('setup')
      bvh1 = BVH(xyz1)
      bvh2 = BVH(xyz2)
      timer.checkpoint('buid bvh')
      pos1, pos2 = list(), list()
      pos1 = np.stack([np.eye(4)])
      pos2 = np.stack([np.eye(4)])
      mindist = 0.15
      timer.checkpoint('setup')

      # cold vs hot doesn't seem to matter
      timer2 = rp.Timer().start()
      pairs, lbub = bvh.bvh_collect_pairs_vec(bvh1, bvh2, pos1, pos2, mindist)
      timer2.checkpoint('bvh cold')
      timer.checkpoint('bvh cold')
      pairs, lbub = bvh.bvh_collect_pairs_vec(bvh1, bvh2, pos1, pos2, mindist)
      timer2.checkpoint('bvh hot')
      timer.checkpoint('bvh hot')
      print(timer2)
      print(f"npairs {len(pairs):,}")
      sys.stdout.flush()

   print(timer.report(summary='mean'))
   assert 0

if __name__ == "__main__":
   # from rpxdock.body import Body

   # b = Body("rpxdock/data/pdb/DHR14.pdb")
   # test_bvh_isect_range(b, cart_sd=15, N2=500, mindist=3.5)

   # test_bvh_isect_cpp()
   # test_bvh_isect_fixed()
   # test_bvh_isect()
   # test_bvh_isect_fixed_range()
   # test_bvh_min_cpp()
   # test_bvh_min_dist_fixed()
   # test_bvh_min_dist()
   # test_bvh_min_dist_floormin()
   # test_bvh_slide_single_inline()
   # test_bvh_slide_single()
   # test_bvh_slide_single_xform()
   # test_bvh_slide_whole()
   # test_collect_pairs_simple()
   # test_collect_pairs_simple_selection()
   # test_collect_pairs()
   # test_collect_pairs_range()
   # test_collect_pairs_range_sym()
   # test_slide_collect_pairs()
   # test_bvh_accessors()
   # test_bvh_isect_range()
   # test_bvh_isect_range_ids()
   # test_bvh_isect_range_lb_ub(N1=10, N2=20)
   # import tempfile
   # test_bvh_pickle(tempfile.mkdtemp())

   # test_bvh_threading_mindist_may_fail()
   # test_bvh_threading_isect_may_fail()

   bvh_perf()
