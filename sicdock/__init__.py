import os

# this was a real bad idea...
# try:
#     from sicdock.util import parallel_build_modules
#
#     parallel_build_modules.parallel_build_modules()
# except:
#     pass

from sicdock import app

# from sicdock import body
from sicdock import bvh
from sicdock import cluster

# from sicdock import data
from sicdock import dockspec
from sicdock import geom

# from sicdock import io
from sicdock import motif
from sicdock import phmap

# from sicdock import rosetta  # pyrosetta slow
# from sicdock import rotamer
from sicdock import sampling
from sicdock import search
from sicdock import sym
from sicdock import util
from sicdock import xbin

from sicdock.xbin import Xbin
from sicdock.motif import ResPairData
from sicdock.motif import ResPairScore

rootdir = os.path.dirname(__file__)
