__author__ = """Will Sheffler"""
__email__ = "willsheffler@gmail.com"
__version__ = "0.1"

import os

# os.environ["CC"] = "gcc-7"  # no idea if this works
# os.environ["CXX"] = "g++-7"  # no idea if this works

# # tell cppimport to say what it's doing
# import cppimport
# cppimport.set_quiet(False)

# set MKL threads to 1
import numpy, ctypes
if hasattr(numpy.__config__, 'mkl_info'):
   mkl_rt = ctypes.CDLL('libmkl_rt.so')
   mkl_rt.mkl_set_num_threads(ctypes.byref(ctypes.c_int(1)))

from rpxdock.util import Bunch, Timer, load, dump
from rpxdock import app
from rpxdock import body
from rpxdock import bvh
from rpxdock import cluster
from rpxdock import data
from rpxdock import fragments
from rpxdock import geom
from rpxdock import io
from rpxdock import motif
from rpxdock import phmap
from rpxdock import rosetta  # dosn't import actual pyrosetta
from rpxdock import rotamer
from rpxdock import sampling
from rpxdock import search
from rpxdock import util
from rpxdock import viz
from rpxdock import xbin

from rpxdock.search import dockspec
from rpxdock.app import options
from rpxdock.body import Body
from rpxdock.bvh import BVH
from rpxdock.data import datadir
from rpxdock.motif import ResPairData, ResPairScore, Xmap
from rpxdock.score import RpxHier
from rpxdock.search import Result, hier_search, grid_search, concat_results
from rpxdock.filter import filter_redundancy
from rpxdock.io import dump_pdb_from_bodies
from rpxdock.geom import symframes
from rpxdock.sampling import ProductHier, ZeroDHier, CompoundHier
from rpxdock.util.cache import GLOBALCACHE, CachedProxy
from rpxdock.xbin import Xbin

rootdir = os.path.dirname(__file__)
