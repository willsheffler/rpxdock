# from rif import rcl
# from rif.rcl import pyrosetta, rosetta
# from rif.chem.biochem import aa_name3s
import os
import numpy as np
import pandas as pd
import xarray

try:
   from functools import lru_cache
except ImportError:
   from backports.functools_lru_cache import lru_cache

boundsfields = "lb1 ub1 lb2 ub2 lb3 ub3 lb4 ub4".split()

def print_full(x):
   pd.set_option("display.max_rows", len(x))
   print(x)
   pd.reset_option("display.max_rows")

def localpath(pth):
   assert not pth.startswith("/")
   return os.path.join(os.path.dirname(__file__), pth)

def get_rotamer_space_raw():
   r = pd.read_csv(localpath("richardson.csv"), header=0, nrows=999, index_col=(0, 1))
   r.reset_index(inplace=True)
   r.sort_values(["aa", "lbl"], inplace=True)
   r.set_index(["aa", "lbl"], inplace=True)
   # r = r.drop('lbl', axis=1)
   # print(r.columns)
   # print(r.iloc[0])
   assert all(x > 0 for x in r.iloc[:, 0])
   for i in range(1, 5):
      r.iloc[:, i] = [float(x) / 100.0 for x in r.iloc[:, i]]
   assert all(0 <= r.f) and all(r.f <= 1.0)
   assert all(0 <= r.fa) and all(r.fa <= 1.0)
   assert all(0 <= r.fb) and all(r.fb <= 1.0)
   assert all(0 <= r.fo) and all(r.fo <= 1.0)
   # print_full(r.x1)
   assert sum(r.x1.isnull()) == 10
   assert all(-180.0 < r.x1.dropna()) and all(r.x1.dropna() <= 180.0)
   # print_full(r.x2)
   assert sum(r.x2.isnull()) == 15
   assert all(-180.0 < r.x2.dropna()) and all(r.x2.dropna() <= 180.0)
   # print_full(r.x3)
   assert sum(r.x3.isnull()) == 62
   assert all(-180.0 < r.x3.dropna()) and all(r.x3.dropna() <= 180.0)
   # print_full(r.x4)
   assert sum(r.x4.isnull()) == 92
   assert all(-180.0 < r.x4.dropna()) and all(r.x4.dropna() <= 180.0)
   assert sum(r.x1r.isnull()) == 10
   assert sum(r.x2r.isnull()) == 15
   # print_full(r.x3r)
   assert sum(r.x3r.isnull()) == 62
   # print_full(r.x4r)
   assert sum(r.x4r.isnull()) == 92
   for i in "1234":
      r["lb" + i] = np.NaN
      r["ub" + i] = np.NaN
      idx = r["x" + i + "r"].notnull()
      r.loc[idx, "lb" + i] = [float(x.split()[0]) for x in r["x" + i + "r"][idx]]
      r.loc[idx, "ub" + i] = [float(x.split()[2]) for x in r["x" + i + "r"][idx]]
   r = r.drop("x1a x2a x3a x4a x1w x2w x3w x4w x1r x2r x3r x4r".split(), axis=1)

   r["nchi"] = 4 - r.loc[:, boundsfields].isnull().sum(axis=1) / 2
   return r

@lru_cache()
def get_rotamer_space(concat=False, disulf=False):
   """B is disulfide, x4 -> x2other
    n num observations
    fabo total fraction/alpha/beta/other
    x#a chi # "act"
    x#  chi # "com"
    x#r chi # range text
    x#w chi # peak width
    lb#/ub# chi # range
    """
   # try:
   # r = pd.read_pickle(localpath("richardson.pkl"))
   # except FileNotFoundError:  # todo: python3 only use FileNotFoundError
   # print('warning: richardson.pkl not available, reading from richardson.csv')
   r = get_rotamer_space_raw()
   # r.to_pickle(localpath('richardson.pkl'))
   # print_full(r.loc[:, 'lb1 ub1 ub2 ub2 lb3 ub3 lb4 ub4'.split()])
   # print_full(r.loc[:, 'x1w x2w x3w x4w'.split()])
   if not disulf:
      r = r.drop("B")
   if concat:
      r = concat_rotamer_space(r)
   r = xarray.Dataset(r)
   r["rotid"] = "dim_0", np.arange(len(r.aa))
   return r

def _roteq(a, b):
   return abs(a % 360.0 - b % 360.0) < 10.0

def merge_on_chi(rs, chi):
   lb = "lb%i" % chi
   ub = "ub%i" % chi
   # print('merge_on_chi')
   rs = rs.copy()  # warnings if I don't do this
   for i in range(10):
      # print('merge_on_chi', i)
      updated = False
      for i, j in ((i, j) for i in range(rs.shape[0]) for j in range(i + 1, rs.shape[0])):
         if _roteq(rs[lb][i], rs[ub][j]):
            # print('++++++++++++++++++++++++ 1 +++++++++++++++++++++++++++')
            # print('lb i, ub j', i, j)
            # print(rs[boundsfields])
            rs.loc[rs.index[i], lb] = rs.loc[rs.index[j], lb]
            # print('drop', j)
            # oldlen = rs.shape[0]
            rs = rs.drop(rs.index[j])
            # print(oldlen, rs.shape[0])
            # print('NEW:')
            # print(rs[boundsfields])
            updated = True
            break
         elif _roteq(rs[ub][i], rs[lb][j]):
            # print('++++++++++++++++++++++++ 2 +++++++++++++++++++++++++++')
            # print('ub i, lb j', i, j)
            # print(rs[boundsfields])
            rs.loc[rs.index[i], ub] = rs.loc[rs.index[j], ub]
            # print('drop', j)
            # oldlen = rs.shape[0]
            rs = rs.drop(rs.index[j])
            # print(oldlen, rs.shape[0])
            # print('NEW:')
            # print(rs[boundsfields])
            updated = True
            break
      if not updated:
         return rs

def concat_rotamer_space(rotspace):
   newdat = []
   for nchi, fixnchi in rotspace.groupby("nchi"):
      keys = (["aa"] + ["lb%i" % i for i in range(1, int(nchi))] +
              ["ub%i" % i for i in range(1, int(nchi))])
      # print(keys)
      for _, fixchiprefix in fixnchi.groupby(keys):
         # print(chiprefix)
         # print(fixchiprefix.loc[:, boundsfields])
         newdat.append(merge_on_chi(fixchiprefix, chi=nchi))
   r = pd.concat(newdat)
   r = r.reset_index()
   r = r.sort_values(["aa", "lbl"])
   r = r.set_index(["aa", "lbl"])
   return r

def sample_rotamer_space(rotspace, resl=[10, 10, 10, 10]):
   """rotamer samples"""
   # for

def get_rotamer_index(rotspace):
   """extract AA structural info via pyrosetta"""
   # print('get_rotamer_coords')
   rcl.init_check()
   chem_manager = rosetta.core.chemical.ChemicalManager
   rts = chem_manager.get_instance().residue_type_set("fa_standard")
   # print(rts)
   rfactory = rosetta.core.conformation.ResidueFactory
   for rname in aa_name3s:
      res = rfactory.create_residue(rts.name_map(rname))
      print(rname, res.nheavyatoms(), res.natoms())
   raise NotImplementedError
   return 1
