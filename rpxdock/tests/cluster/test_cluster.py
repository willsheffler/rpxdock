from rpxdock.cluster import cookie_cutter
from scipy.spatial.distance import pdist

import numpy as np

def test_cluster():
   mesh = np.meshgrid(np.arange(3), np.arange(3), np.arange(3))
   mesh = np.stack(mesh).swapaxes(0, -1).reshape(-1, len(mesh)).astype("f8")

   assert 27 == len(cookie_cutter(mesh, 0.99)[0])
   assert 14 == len(cookie_cutter(mesh, 1.0)[0])
   assert 14 == len(cookie_cutter(mesh, 1.01)[0])
   assert 9 == len(cookie_cutter(mesh, 1.42)[0])
   assert 8 == len(cookie_cutter(mesh, 1.74)[0])
   assert 5 == len(cookie_cutter(mesh, 2)[0])
   assert 4 == len(cookie_cutter(mesh, 2.26)[0])
   assert 2 == len(cookie_cutter(mesh, 2.83)[0])
   assert 1 == len(cookie_cutter(mesh, 3.48)[0])

def test_cluster_rand():
   thresh = 0.2
   npts = 200
   ncol = 4
   minmindis = 9e9
   nhit = list()

   for i in range(1000):
      x = np.random.random((npts, ncol))
      keep, clustid = cookie_cutter(x, thresh)
      y = x[keep]
      # print(x.shape, y.shape[0], thresh)
      nhit.append(y.shape[0])
      mindis = np.min(pdist(y))
      minmindis = min(mindis, minmindis)
      assert mindis > thresh

      big_cluster = np.argmax(np.bincount(clustid))
      clust_points = x[clustid == big_cluster]
      p1 = clust_points[0]
      for p2 in clust_points[1:]:
         d = np.sqrt(np.sum((p1 - p2)**2))
         assert d <= thresh

   assert minmindis < thresh * 1.1  # arbitrary, failure should be really rare

if __name__ == "__main__":
   test_cluster()
   test_cluster_rand()
