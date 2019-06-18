import numpy as np
from rpxdock.geom import primitive_test
from rpxdock.geom import miniball_test, miniball
from rpxdock.geom import xform_dist2
import rpxdock.homog as hm

def test_geom_sphere():
   assert primitive_test.TEST_geom_primitive_sphere()

def test_geom_welzl_sphere():
   assert primitive_test.TEST_geom_primitive_welzl_bounding_sphere()

def test_miniball_cpp():
   assert miniball_test(10_000, 7, False)
   assert miniball_test(10_000, 12, False)

def test_miniball_py():
   crd = np.random.randn(10_000, 7)
   sph = miniball(crd)
   rad, *cen = sph
   d = np.linalg.norm(crd - cen, axis=1)
   print(np.min(d), np.max(d))
   assert np.max(d) <= rad + 0.00000001

def test_xform_dist():
   N = 1
   N2 = 10
   for i in range(N):
      lever = np.random.rand() * 100
      ang = np.random.rand(N2) * 90
      ang
      xa = np.eye(4)
      xb = hm.hrot(np.random.randn(N2, 3), ang, degrees=True)
      d = xform_dist2(xa, xb, lever)
      approx_degrees = np.sqrt(d) * 180 / np.pi / lever
      assert np.all(0.95 < approx_degrees / ang)
      assert np.all(approx_degrees / ang <= 1.0)

      assert np.all(xform_dist2(xb, xa, 1).T == xform_dist2(xa, xb, 1))

   xa = hm.htrans(np.random.randn(10, 3))
   xb = hm.htrans(np.random.randn(20, 3))
   d2 = xform_dist2(xa, xb, 1)
   trans = xa[:, None, :3, 3] - xb[:, :3, 3]
   dcart = np.linalg.norm(trans, axis=-1)
   assert np.allclose(np.sqrt(d2), dcart)

if __name__ == "__main__":
   # test_geom_sphere()
   # test_geom_welzl_sphere()
   # test_miniball_cpp()
   # test_miniball_py()
   test_xform_dist()
