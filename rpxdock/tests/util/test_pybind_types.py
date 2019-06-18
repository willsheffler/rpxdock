from cppimport import import_hook
from rpxdock.util import pybind_types_test
import numpy as np

def test_xform_round_trip():
   # debug = [80, 81, 82, 83, 10, 11, 12, 13, 20, 21, 22, 23, 31, 32, 33, 34]
   x = np.array([np.eye(4)] * 10).reshape(-1, 4, 4)
   y = pybind_types_test.test_xform_round_trip(x)
   assert x.shape == y.shape
   assert np.all(x == y)

   a, b = id(x[0, 0, 0]), id(y[0, 0, 0])  # no copy
   assert a == b
   assert np.allclose(x[3, 1, 3], 1)
   assert np.allclose(x[7, 1, 2], 9)

def test_xform_round_trip_2d():
   x = np.eye(4)
   y = pybind_types_test.test_xform_round_trip(x)
   assert x.shape == (4, 4)
   assert y.shape == (1, 4, 4)
   assert np.all(x == y)
   a, b = id(x[0, 0]), id(y[0, 0, 0])  # no copy
   assert a == b
   assert np.allclose(x[1, 3], 1)
   assert np.allclose(x[1, 2], 9)
   assert np.allclose(y[0, 1, 3], 1)
   assert np.allclose(y[0, 1, 2], 9)

if __name__ == "__main__":
   test_xform_round_trip()
   test_xform_round_trip_2d()
