import _pickle
from time import perf_counter
import numpy as np
from cppimport import import_hook
import rpxdock.homog as hm
from rpxdock.bvh.bvh_nd import *
from scipy.spatial.distance import cdist

def test_bvh_bvh_isect7():
   nhit, ntot = 0, 0
   for iter in range(1):
      pts = np.random.randn(1_000, 7) + [2, 2, 2, 2, 2, 2, 2]
      bvh = create_bvh7d(pts)
      assert np.allclose(bvh.com(), np.mean(pts, axis=0))
      pts2 = np.random.randn(1_000, 7)
      bvh2 = create_bvh7d(pts2)
      for mindis in np.arange(0.1, 2, 0.1):
         isect1 = bvh_bvh_isect7d(bvh, bvh2, mindis)
         isect2 = bvh_bvh_isect7d_naive(bvh, bvh2, mindis)
         assert isect1 == isect2
         nhit += isect1
         ntot += 1
   print("frac", nhit / ntot)

def test_bvh_isect7():
   Nrep, Nbvh, Nsamp = 1, 1_000, 10000
   N = Nrep * Nsamp
   tbvh, tnai = 0, 0
   nhit, ntot = 0, 0
   for i in range(Nrep):
      pts_bvh = np.random.rand(Nbvh, 7)
      bvh = create_bvh7d(pts_bvh)
      assert np.allclose(bvh.com(), np.mean(pts_bvh, axis=0))

      mindis = 0.2
      samp = np.random.rand(Nsamp, 7)
      tbvh0 = perf_counter()
      isect1 = 0 <= bvh_isect7d(bvh, samp, mindis)
      tbvh += perf_counter() - tbvh0
      tnai0 = perf_counter()
      isect2 = 0 <= bvh_isect7d_naive(bvh, samp, mindis)
      tnai += perf_counter() - tnai0
      assert np.all(isect1 == isect2)
      nhit += np.sum(isect1)
      ntot += Nsamp

   print(
      f" bvh_isect7d frchit: {nhit/ntot:4.3f}",
      f"bvh rate: {int(N/tbvh):,} naive rate: {int(N/tnai):,}",
   )

def test_bvh_mindist4():
   Nrep, Nbvh, Nsamp = 1, 1_000, 1000
   N = Nrep * Nsamp
   tbvh, tnai = 0, 0
   mindis = 9e9
   for i in range(Nrep):
      pts_bvh = np.random.rand(Nbvh, 4)
      bvh = create_bvh4d(pts_bvh)
      assert np.allclose(bvh.com(), np.mean(pts_bvh, axis=0))

      samp = np.random.rand(Nsamp, 4) + [0.2, 0.4, 0, 0]
      tbvh0 = perf_counter()
      mindist1, w1 = bvh_mindist4d(bvh, samp)
      tbvh += perf_counter() - tbvh0
      tnai0 = perf_counter()
      mindist2, w2 = bvh_mindist4d_naive(bvh, samp)
      tnai += perf_counter() - tnai0
      assert np.allclose(mindist1, mindist2)
      assert np.all(w1 == w2)
      mindis = min(mindis, np.min(mindist1))

   print(
      f" bvh_mindist4d mind: {mindis:5.3f} bvh rate: {int(N/tbvh):,} naive rate: {int(N/tnai):,}",
      # f"{tnai/tbvh:7.3f}",
   )
   print("bvh", tbvh)
   print("nai", tnai)

def test_bvh_mindist7():
   Nrep, Nbvh, Nsamp = 1, 1_000, 10000
   N = Nrep * Nsamp
   tbvh, tnai = 0, 0
   mindis = 9e9
   for i in range(Nrep):
      pts_bvh = np.random.rand(Nbvh, 7)
      bvh = create_bvh7d(pts_bvh)
      assert np.allclose(bvh.com(), np.mean(pts_bvh, axis=0))

      samp = np.random.rand(Nsamp, 7) + [0, 0, 0, 0, 0, 0, 0]
      tbvh0 = perf_counter()
      mindist1, w1 = bvh_mindist7d(bvh, samp)
      tbvh += perf_counter() - tbvh0
      tnai0 = perf_counter()
      mindist2, w2 = bvh_mindist7d_naive(bvh, samp)
      tnai += perf_counter() - tnai0
      assert np.allclose(mindist1, mindist2)
      assert np.all(w1 == w2)
      mindis = min(mindis, np.min(mindist1))

   print(
      f" bvh_mindist7d mind: {mindis:5.3f} bvh rate: {int(N/tbvh):,} naive rate: {int(N/tnai):,}",
      # f"{tnai/tbvh:7.3f}",
   )
   print("bvh", tbvh)
   print("nai", tnai)

if __name__ == "__main__":
   # print(4096 / 2.9, 1000000 / 38.3)
   # test_bvh_bvh_isect7()
   # test_bvh_isect7()
   test_bvh_mindist7()
   # test_bvh_mindist4()
