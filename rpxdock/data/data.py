import os, pickle
from functools import lru_cache
import rpxdock as rp

datadir = os.path.dirname(__file__)
pdbdir = os.path.join(os.path.dirname(__file__), "pdb")
bodydir = os.path.join(os.path.dirname(__file__), "body")
hscoredir = os.path.join(os.path.dirname(__file__), "hscore")
testdatadir = os.path.join(os.path.dirname(__file__), "testdata")

@lru_cache()
def get_test_data(name):
   print('!!', os.path.join(testdatadir, f'{name}.pickle'))
   with open(os.path.join(testdatadir, f'{name}.pickle'), 'rb') as inp:
      return pickle.load(inp)

@lru_cache()
def get_body(name):
   fname = os.path.join(bodydir, name + '.pickle')
   print(f'get_body("{fname}")')
   with open(fname, 'rb') as inp:
      return pickle.load(inp)

@lru_cache()
def small_respairdat():
   from rpxdock import ResPairData
   with open(os.path.join(datadir, "respairdat10_plus_xmap_rots.pickle"), "rb") as inp:
      return ResPairData(pickle.load(inp))

@lru_cache()
def small_respairscore():
   with open(os.path.join(datadir, "pairscore10.pickle"), "rb") as inp:
      return pickle.load(inp)

@lru_cache()
def small_hscore():
   from rpxdock import RpxHier  # avoid cyclic import
   return RpxHier('small_ilv_h', hscore_data_dir=hscoredir)
