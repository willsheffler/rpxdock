import os, _pickle
from functools import lru_cache

datadir = os.path.dirname(__file__)
pdbdir = os.path.join(os.path.dirname(__file__), "pdb")
bodydir = os.path.join(os.path.dirname(__file__), "body")
hscoredir = os.path.join(os.path.dirname(__file__), "hscore")
testdatadir = os.path.join(os.path.dirname(__file__), "testdata")

@lru_cache()
def small_respairdat():
   from rpxdock import ResPairData
   with open(os.path.join(datadir, "respairdat10_plus_xmap_rots.pickle"), "rb") as inp:
      return ResPairData(_pickle.load(inp))

@lru_cache()
def small_respairscore():
   with open(os.path.join(datadir, "pairscore10.pickle"), "rb") as inp:
      return _pickle.load(inp)

@lru_cache()
def small_hscore():
   from rpxdock import HierScore  # avoid cyclic import
   return HierScore('small_ilv_h', hscore_data_dir=hscoredir)

@lru_cache()
def body_dhr14():
   with open(os.path.join(bodydir, 'dhr14.pickle'), 'rb') as inp:
      return _pickle.load(inp)

@lru_cache()
def body_dhr64():
   with open(os.path.join(bodydir, 'dhr64.pickle'), 'rb') as inp:
      return _pickle.load(inp)

@lru_cache()
def body_small_c3_hole():
   with open(os.path.join(bodydir, 'small_c3_hole.pickle'), 'rb') as inp:
      return _pickle.load(inp)

@lru_cache()
def body_c3_mono():
   with open(os.path.join(bodydir, 'test_c3_mono.pickle'), 'rb') as inp:
      return _pickle.load(inp)

@lru_cache()
def get_test_data(name):
   with open(os.path.join(testdatadir, f'{name}.pickle'), 'rb') as inp:
      return _pickle.load(inp)
