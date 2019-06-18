import sys
import gzip
import numpy as np
import rpxdock.sampling.orientations as ori
import pytest

from cppimport import import_hook
from rpxdock.sampling._orientations_test import *

def test_orientation_cpp():
   assert TEST_Orientation_read_karney_datasets()

def test_read_karney_orientations():
   with gzip.open(ori.karney_data_path("c48u1.grid.gz")) as input:
      if sys.version_info.major is 3:
         quat, weight = ori.read_karney_orientations(str(input.read(), "utf-8"))
      else:
         quat, weight = ori.read_karney_orientations(str(input.read()))
   norms = np.linalg.norm(quat, axis=1)
   assert np.all(np.abs(1.0 - norms) < 0.0001)
   assert np.min(weight) == np.max(weight) == 1.0

def test_karney_oddball():
   with pytest.raises(IOError):
      ori.quaternion_set_by_name("foobar")

def test_karney_10():
   quat, weight = ori.quaternion_set_with_covering_radius_degrees(10)
   assert len(quat) == 7416
   print(weight[:10])
   assert abs(1.0 - weight.mean()) < 0.00001

def test_karney_lengths_weights():
   # only check smallest 20 for speed
   for name in ori._karney_index.name[:20]:
      quat, weight = ori.quaternion_set_by_name(name)
      assert len(weight)
      assert abs(1.0 - weight.mean()) < 1e-6
      norms = np.linalg.norm(quat, axis=1)
      assert np.all(np.abs(1.0 - norms) < 0.0001)

def test_karney_by_covrad():
   assert ori.karney_name_by_radius(100) == "c48u1"
   assert ori.karney_name_by_radius(30) == "c48u27"
   assert ori.karney_name_by_radius(20) == "c48u83"
   assert ori.karney_name_by_radius(10) == "c48n309"
   assert ori.karney_name_by_radius(8) == "c48u815"
   assert ori.karney_name_by_radius(5) == "c48u2947"
   assert ori.karney_name_by_radius(2) == "c48u40003"
   assert ori.karney_name_by_radius(1) == "c48u295333"
   assert ori.karney_name_by_radius(0) == "c48u312831"

# def test_filter_by_axis():
#     q, w = ori.quaternion_set_with_covering_radius_degrees(100)
#     print q.shape
#     q_filt = ori.filter_quaternion_set_axis_within(
#     q, np.array((1, 0, 0)), 90.0)
#     assert len(q_filt) < len(q)
