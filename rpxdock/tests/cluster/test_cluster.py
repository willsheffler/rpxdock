from rpxdock.cluster import cookie_cutter
from scipy.spatial.distance import pdist

import numpy as np

def test_cluster():
   mesh = np.meshgrid(np.arange(3), np.arange(3), np.arange(3))
   mesh = np.stack(mesh).swapaxes(0, -1).reshape(-1, len(mesh)).astype("f8")

   assert 27 == len(cookie_cutter(mesh, 0.99))
   assert 14 == len(cookie_cutter(mesh, 1.0))
   assert 14 == len(cookie_cutter(mesh, 1.01))
   assert 9 == len(cookie_cutter(mesh, 1.42))
   assert 8 == len(cookie_cutter(mesh, 1.74))
   assert 5 == len(cookie_cutter(mesh, 2))
   assert 4 == len(cookie_cutter(mesh, 2.26))
   assert 2 == len(cookie_cutter(mesh, 2.83))
   assert 1 == len(cookie_cutter(mesh, 3.48))

def test_cluster_rand():
   thresh = 0.1
   npts = 100
   ncol = 4
   minmindis = 9e9
   nhit = list()

   for i in range(1000):
      x = np.random.random((npts, ncol))
      keep = cookie_cutter(x, thresh)
      y = x[keep]
      # print(x.shape, y.shape[0], thresh)
      nhit.append(y.shape[0])
      mindis = np.min(pdist(y))
      minmindis = min(mindis, minmindis)
      assert mindis > thresh

   assert minmindis < thresh * 1.1  # arbitrary, failure should be really rare

if __name__ == "__main__":
   test_cluster_rand()
