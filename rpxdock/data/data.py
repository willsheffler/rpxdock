import os, _pickle
from functools import lru_cache

datadir = os.path.dirname(__file__)
pdbdir = os.path.join(os.path.dirname(__file__), "pdb")
bodydir = os.path.join(os.path.dirname(__file__), "body")
hscoredir = os.path.join(os.path.dirname(__file__), "hscore")
testdatadir = os.path.join(os.path.dirname(__file__), "testdata")
paramsdir = os.path.join(os.path.dirname(__file__), "rosetta_params")

def rosetta_params_files():
   return [paramsdir + '/' + f for f in os.listdir(paramsdir) if f.endswith('.params')]

def rosetta_patch_files():
   return [paramsdir + '/' + f for f in os.listdir(paramsdir) if f.endswith('.txt')]

@lru_cache()
def get_test_data(name):
   from rpxdock.search import result_from_tarball
   if name.endswith('.pickle'):
      with open(os.path.join(testdatadir, name), 'rb') as inp:
         return _pickle.load(inp)
   else:
      try:
         return result_from_tarball(os.path.join(testdatadir, f'{name}.result.txz'))
      except FileNotFoundError:
         print('warning: using old pickle format for test result')
         with open(os.path.join(testdatadir, f'{name}.pickle'), 'rb') as inp:
            return _pickle.load(inp)

@lru_cache()
def get_body_cached(name):
   return get_body_copy(name)

get_body = get_body_cached

def get_body_copy(name):
   fname = os.path.join(bodydir, name + '.pickle')
   print(f'get_body_copy("{fname}")')
   with open(fname, 'rb') as inp:
      return _pickle.load(inp)

@lru_cache()
def small_respairdat():
   import xarray as xr, rpxdock as rp
   fn = os.path.join(datadir, "respairdat10_plus_xmap_rots.nc")
   # with open(fn, "rb") as inp:
   #   return ResPairData(_pickle.load(inp))
   with xr.open_dataset(fn, decode_cf=True) as rpd:
      rpd.load()
      return rp.motif.pairdat.ResPairData(rpd)

@lru_cache()
def small_respairscore():
   # with open(os.path.join(datadir, "pairscore10.pickle"), "rb") as inp:
   # return _pickle.load(inp)
   from rpxdock.util import load
   return load(os.path.join(datadir, "pairscore10.rpx.txz"))

@lru_cache()
def small_hscore():
   from rpxdock import RpxHier  # avoid cyclic import
   return RpxHier('small_ilv_h', hscore_data_dir=hscoredir)
