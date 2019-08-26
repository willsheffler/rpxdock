import os, _pickle
from functools import lru_cache

datadir = os.path.dirname(__file__)
pdbdir = os.path.join(os.path.dirname(__file__), "pdb")
bodydir = os.path.join(os.path.dirname(__file__), "body")
hscoredir = os.path.join(os.path.dirname(__file__), "hscore")
testdatadir = os.path.join(os.path.dirname(__file__), "testdata")

@lru_cache()
def get_test_data(name):
   with open(os.path.join(testdatadir, f'{name}.pickle'), 'rb') as inp:
      return _pickle.load(inp)

@lru_cache()
def get_body(name):
   with open(os.path.join(bodydir, name + '.pickle'), 'rb') as inp:
      return _pickle.load(inp)

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
   from rpxdock import RpxHier  # avoid cyclic import
   return RpxHier('small_ilv_h', hscore_data_dir=hscoredir)
