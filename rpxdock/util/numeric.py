import numpy as np

def svd(x):
   U, Sigma, Vh = np.linalg.svd(x, full_matrices=False, compute_uv=True)
   X_svd = np.dot(U, np.diag(Sigma))
   return X_svd

def pca_eig(coords):
   n = len(coords)
   x = coords - coords.mean(axis=0)
   cov = np.dot(x.T, x) / (n - 1)
   e, evec = np.linalg.eig(cov)
   idx = np.argsort(-np.abs(e))
   return e[idx], evec[idx]

def pca(coords):
   eigen_vals, eigen_vecs = pca_eig(coords)
   return np.dot(x, eigen_vecs)
