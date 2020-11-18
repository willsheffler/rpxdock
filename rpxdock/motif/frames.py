import os, sys, time, _pickle
from cppimport import import_hook
import numpy as np, xarray as xr

from rpxdock.xbin import xbin_util as xu
from rpxdock.rotamer import get_rotamer_space, assign_rotamers, check_rotamer_deviation
from rpxdock.motif import _motif as cpp
from rpxdock.motif.pairdat import ResPairData
from rpxdock.data import pdbdir

# [ 0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19]
# ['A' 'C' 'D' 'E' 'F' 'G' 'H' 'I' 'K' 'L' 'M' 'N' 'P' 'Q' 'R' 'S' 'T' 'V' 'W' 'Y']
# [0 1 2]
# ['E' 'H' 'L']

def ss_to_ssid(ss):
   ssid = np.zeros(ss.shape, dtype="u8")
   ssid[ss == "E"] = 0
   ssid[ss == "H"] = 1
   ssid[ss == "L"] = 2
   return ssid

def _convert_point(point):
   if not isinstance(point, (np.ndarray, xr.DataArray)):
      point = np.array([[point[0], point[1], point[2], 1]])
   if isinstance(point, xr.DataArray):
      return point.data
   return point

def stub_from_points(cen, pa=None, pb=None, dtype="f4"):
   cen = _convert_point(cen)
   pa = _convert_point(pa)
   pb = _convert_point(pb)
   assert len(cen) == len(pa) == len(pb)
   assert cen.ndim == pa.ndim == pb.ndim == 2
   stub = np.zeros((len(cen), 4, 4), dtype=dtype)
   stub[:, 3, 3] = 1
   e1 = cen[:, :3] - pa[:, :3]
   e1 /= np.linalg.norm(e1, axis=1)[:, None]
   e3 = np.cross(e1, pb[:, :3] - pa[:, :3])
   e3 /= np.linalg.norm(e3, axis=1)[:, None]
   e2 = np.cross(e3, e1)
   stub[:, :3, 0] = e1
   stub[:, :3, 1] = e2
   stub[:, :3, 2] = e3
   stub[:, :3, 3] = cen[:, :3]
   assert np.allclose(np.linalg.det(stub), 1)
   return stub

def bb_stubs(n, ca=None, c=None, dtype="f4"):
   if ca is None:
      assert n.ndim == 3
      assert n.shape[1] >= 3  # n, ca, c
      ca = n[:, 1, :3]
      c = n[:, 2, :3]
      n = n[:, 0, :3]

   n = _convert_point(n)
   ca = _convert_point(ca)
   c = _convert_point(c)

   assert len(n) == len(ca) == len(c)
   assert n.ndim == ca.ndim == c.ndim == 2
   stub = np.zeros((len(n), 4, 4), dtype=dtype)
   stub[:, 3, 3] = 1
   e1 = n[:, :3] - ca[:, :3]
   e1 /= np.linalg.norm(e1, axis=1)[:, None]
   e3 = np.cross(e1, c[:, :3] - ca[:, :3])
   e3 /= np.linalg.norm(e3, axis=1)[:, None]
   e2 = np.cross(e3, e1)
   stub[:, :3, 0] = e1
   stub[:, :3, 1] = e2
   stub[:, :3, 2] = e3
   # magic numbers from rosetta centroids in some set of pdbs
   avg_centroid_offset = [-0.80571551, -1.60735769, 1.46276045]
   t = stub[:, :3, :3] @ avg_centroid_offset + ca[:, :3]
   stub[:, :3, 3] = t
   assert np.allclose(np.linalg.det(stub), 1)
   return stub

def get_pair_keys(rp, xbin, min_pair_score, min_ssep, use_ss_key, **kw):
   mask = (rp.p_resj - rp.p_resi).data >= min_ssep
   mask = np.logical_and(mask, -rp.p_etot.data >= min_pair_score)
   resi = rp.p_resi.data[mask]
   resj = rp.p_resj.data[mask]
   stub = rp.stub.data
   kij = np.zeros(len(rp.p_resi), dtype="u8")
   kji = np.zeros(len(rp.p_resi), dtype="u8")
   if use_ss_key:
      ss = rp.ssid.data
      assert np.max(ss) <= 2
      assert np.min(ss) >= 0
      assert resi.dtype == ss.dtype
      kij[mask] = xu.sskey_of_selected_pairs(xbin, resi, resj, ss, ss, stub, stub)
      kji[mask] = xu.sskey_of_selected_pairs(xbin, resj, resi, ss, ss, stub, stub)
   else:
      kij[mask] = xu.key_of_selected_pairs(xbin, resi, resj, stub, stub)
      kji[mask] = xu.key_of_selected_pairs(xbin, resj, resi, stub, stub)
   return kij, kji

def add_xbin_to_respairdat(rp, xbin, **kw):
   kij, kji = get_pair_keys(rp, xbin, **kw)
   rp.data["kij"] = ["pairid"], kij
   rp.data["kji"] = ["pairid"], kji
   rp.attrs["xbin_type"] = "wtihss"

def add_rots_to_respairdat(rp, rotspace, **kw):
   rotids, rotlbl, rotchi = assign_rotamers(rp, rotspace)
   rp.data["rotid"] = ["resid"], rotids
   rp.data.attrs["rotlbl"] = rotlbl
   rp.data.attrs["rotchi"] = rotchi

def make_respairdat_subsets(rp):
   keep = np.arange(len(rp.pdbid))
   np.random.shuffle(keep)
   rp10 = rp.subset_by_pdb(keep[:10])
   with open("rpxdock/tests/motif/respairdat10_plus_xmap_rots.pickle", "wb") as out:
      _pickle.dump(rp10.data, out)
   rp100 = rp.subset_by_pdb(keep[:100])
   with open("rpxdock/tests/motif/respairdat100.pickle", "wb") as out:
      _pickle.dump(rp100.data, out)
   rp1000 = rp.subset_by_pdb(keep[:1000])
   with open("rpxdock/tests/motif/respairdat1000.pickle", "wb") as out:
      _pickle.dump(rp1000.data, out)

def remove_redundant_pdbs(pdbs, sequence_identity=30):
   assert sequence_identity in (30, 40, 50, 70, 90, 95, 100)
   listfile = "pdbids_20190403_si%i.txt" % sequence_identity
   with open(os.path.join(pdbdir, listfile)) as inp:
      goodids = set(l.strip() for l in inp.readlines())
      assert all(len(g) == 4 for g in goodids)
   return np.array([i for i, p in enumerate(pdbs) if p[:4].upper() in goodids])
