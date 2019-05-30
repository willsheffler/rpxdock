import os, _pickle
from functools import lru_cache

datadir = os.path.dirname(__file__)
pdbdir = os.path.join(os.path.dirname(__file__), "pdb")

@lru_cache()
def small_respairdat():
   from sicdock.motif import ResPairData
   with open(os.path.join(datadir, "respairdat10_plus_xmap_rots.pickle"), "rb") as inp:
      return ResPairData(_pickle.load(inp))

@lru_cache()
def small_respairscore():
   from sicdock.motif import ResPairData
   with open(os.path.join(datadir, "pairscore10.pickle"), "rb") as inp:
      return _pickle.load(inp)
