from rpxdock.util.numeric import *

def test_pca():
   x = np.random.randn(10000, 3) * [1, 2, 3]
   e, evec = pca_eig(x)
   print(e)
   print(evec)
   assert np.allclose(1, np.linalg.norm(evec, axis=1))
   assert abs(evec[0, 2]) > 0.99
   assert abs(evec[1, 1]) > 0.99
   assert abs(evec[2, 0]) > 0.99
   assert abs(e[0] - 9) < 0.5
   assert abs(e[1] - 4) < 0.3
   assert abs(e[2] - 1) < 0.1

def test_eig_svd():
   n = 1000
   x = np.random.randn(n, 3) * [1, 2, 3]

   cov = np.dot(x.T, x) / (n - 1)
   eigen_vals, eigen_vecs = np.linalg.eig(cov)

   u, s, v = np.linalg.svd(x, full_matrices=False)
   svd_eig = s**2 / (n - 1)

   assert np.allclose(sorted(svd_eig), sorted(eigen_vals))

if __name__ == "__main__":
   test_pca()
   test_eig_svd()
