import sys
import os
from os.path import join
import gzip

import pandas as pd

from cppimport import import_hook
from rpxdock.sampling._orientations import read_karney_orientations

if sys.version_info[0] < 3:
   from io import BytesIO

   StrIO = BytesIO
else:
   from io import StringIO

   StrIO = StringIO

_KARNEY_PATH = os.path.join(os.path.dirname(__file__), "data", "karney")

with open(os.path.join(_KARNEY_PATH, "index.dat")) as fin:
   karney_index_str = fin.read()
_karney_index = pd.read_csv(StrIO(karney_index_str), sep="\s+")

def karney_data_path(fname):
   # print("_KARKEY_PATH:", _KARNEY_PATH)
   return join(_KARNEY_PATH, fname)

def quats_from_karney_file(fname):
   with gzip.open(fname) as input:
      if sys.version_info.major is 3:
         quat, weight = read_karney_orientations(str(input.read(), "utf-8"))
      else:
         quat, weight = read_karney_orientations(str(input.read()))
   return quat, weight

def karney_name_by_radius(cr):
   i = sum(_karney_index.radius > cr)
   if i == _karney_index.shape[0]:
      i -= 1
   return _karney_index.iloc[i, 0]

def quaternion_set_with_covering_radius_degrees(cr=63):
   # print(os.getcwd())
   fname = karney_data_path(karney_name_by_radius(cr) + ".grid.gz")
   return quats_from_karney_file(fname)

def quaternion_set_by_name(name):
   fname = karney_data_path(name + ".grid.gz")
   if name not in _karney_index.name.values:
      raise IOError("unknown karney file " + name)
   return quats_from_karney_file(fname)

def filter_quaternion_set_axis_within(quats, axis, angle):
   raise NotImplementedError
   return quats
