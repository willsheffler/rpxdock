import os, _pickle
import numpy as np
from rpxdock.util import Bunch
from rpxdock.phmap import PHMap_u8u8, PHMap_u8f8
from rpxdock.motif import bb_stubs, add_xbin_to_respairdat
from rpxdock.motif import add_rots_to_respairdat, get_pair_keys
from rpxdock.motif import jagged_bin, logsum_bins
from rpxdock.xbin import Xbin
from rpxdock.rotamer import get_rotamer_space, assign_rotamers

# def load_respairscore(path):
#     assert os.path.isdir(path)
#     with open(os.path.join(path, "resdata.pickle"), "rb") as inp:
#         rps = _pickle.load(inp)
#     rps.score_map = PHMap_u8f8()
#     rps.range_map = PHMap_u8u8()
#     rps.score_map.load(os.path.join(path, "score_map.bin"))
#     rps.range_map.load(os.path.join(path, "range_map.bin"))
#     return rps

class Xmap:
   def __init__(self, xbin, phmap, attrs={}, rehash_bincens=False):
      self.xbin = xbin
      self.phmap = phmap
      self.attr = Bunch(attrs)
      if rehash_bincens:
         k, v = self.phmap.items_array()
         self.phmap[xbin.key_of(xbin.bincen_of(k))] = v

   def __len__(self):
      return len(self.phmap)

   def __eq__(self, other):
      return self.xbin == other.xbin and self.phmap == other.phmap

   def key_of(self, x_or_k):
      if x_or_k.dtype == np.uint64:
         return x_or_k
      else:
         return self.xbin.key_of(x_or_k)

   def __getitem__(self, x_or_k):
      return self.phmap[self.key_of(x_or_k)]

   def has(self, x_or_k):
      return self.phmap.has(self.key_of(x_or_k))

   def __contains__(self, x_or_k):
      return self.phmap.__contains__(self.key_of(x_or_k))

   def keys(self, n=-1):
      return self.phmap.keys(n)

   def xforms(self, n=-1):
      return self.xbin.bincen_of(self.phmap.keys(n))

class ResPairScore:
   def __init__(self, xbin, keys, score_map, range_map, res1, res2, rotspace, rp):
      assert np.all(score_map.has(keys))
      assert np.all(range_map.has(keys))
      self.xbin = xbin
      self.rotspace = rotspace
      self.keys = keys
      self.score_map = score_map
      self.range_map = range_map
      self.respair = np.stack([res1.astype("i4"), res2.astype("i4")], axis=1)
      self.aaid = rp.aaid.data.astype("u1")
      self.ssid = rp.ssid.data.astype("u1")
      self.rotid = rp.rotid.data
      self.stub = rp.stub.data.astype("f4")
      self.pdb = rp.pdb.data[rp.r_pdbid.data]
      self.resno = rp.resno.data
      self.rotchi = rp.rotchi
      self.rotlbl = rp.rotlbl
      self.id2aa = rp.id2aa.data
      self.id2ss = rp.id2ss.data
      self.hier_maps = []
      self.hier_resls = np.array([])
      self.attr = Bunch()

   def add_hier_score(self, resl, scoremap):
      self.hier_resls = np.append(self.hier_resls, resl)
      self.hier_maps.append(scoremap)

   def hier_score(self, resl):
      w = np.which(self.hier_resls)[0]
      if np.abs(1 - resl / self.hier_resls[w]) > 0.1:
         return None
      print(w)
      return self.hier_maps[w]

   def bin_score(self, keys):
      score = np.zeros(len(keys))
      mask = self.score_map.has(keys)
      score[mask] = self.score_map[keys[mask]]
      return score

   def bin_get_all_data(self, keys):
      mask = self.range_map.has(keys)
      ranges = self.range_map[keys[mask]].view("u4").reshape(-1, 2)
      out = np.empty(len(ranges), dtype="O")
      for i, r in enumerate(ranges):
         lb, ub = r
         res = self.respair[lb:ub]
         rots = self.rotid[res]
         aas = self.aaid[res]
         stubs = self.stub[res]
         pdbs = self.pdb[res[:, 0]]
         resno = self.resno[res]
         # print(aas.shape, rots.shape, stubs.shape, pdbs.shape, resno.shape)
         out[i] = pdbs, resno, aas, rots, stubs
      return out, mask

   # def dump(self, path):
   #     if os.path.exists(path):
   #         assert os.path.isdir(path)
   #     else:
   #         os.mkdir(path)
   #     self.score_map.dump(os.path.join(path, "score_map.bin"))
   #     self.range_map.dump(os.path.join(path, "range_map.bin"))
   #     tmp = self.score_map, self.range_map
   #     self.score_map, self.range_map = None, None
   #     with open(os.path.join(path, "resdata.pickle"), "wb") as out:
   #         _pickle.dump(self, out)
   #     self.score_map, self.range_map = tmp

   def bin_respairs(self, key):
      r = self.rangemap[k]
      lb = np.right_shift(r, 32)
      ub = np.right_shift(np.left_shift(r), 32)
      return self.respair[lb:ub]

   def __str__(self):
      return (f"ResPairScore: npdb {len(self.pdb):,} nres {len(self.aaid):,} " +
              f"npair {len(self.keys):,}\n   base Xbin: " +
              f"cart_resl {self.xbin.cart_resl:5.2f} " + f"ori_resl {self.xbin.ori_resl:6.2f}" +
              f" max_cart {self.xbin.max_cart:7.2f} \n        Xmap: " +
              f"score_map {len(self.score_map):,} " + f"range_map {len(self.range_map):,}")

def create_res_pair_score_map(rp, xbin, min_bin_score, **kw):
   kij, kji = get_pair_keys(rp, xbin, **kw)
   keys0 = np.concatenate([kij, kji])
   order, binkey, binrange = jagged_bin(keys0)
   assert len(binkey) == len(binrange)
   epair = np.concatenate([rp.p_etot.data, rp.p_etot.data])[order]
   ebin = logsum_bins(binrange, -epair)

   assert len(ebin) == len(binkey)
   mask = ebin > min_bin_score
   # print(
   # np.sum(mask), "nbin/nkey", len(binkey) / len(keys0), "mean ebin", np.mean(ebin)
   # )
   pair_score = PHMap_u8f8()
   pair_score[binkey[mask]] = ebin[mask]
   xmap = Xmap(xbin, pair_score, rehash_bincens=True)
   xmap.attr.min_bin_score = min_bin_score
   xmap.attr.min_pair_score = kw["min_pair_score"]
   xmap.attr.use_ss_key = kw["use_ss_key"]
   xmap.attr.kw = kw
   return xmap, order, binkey[mask], binrange[mask]

def create_res_pair_score(rp, xbin, **kw):
   rotspace = get_rotamer_space()
   if "stub" not in rp.data:
      rp.data["stub"] = ["resid", "hrow", ""], bb_stubs(rp.n, rp.ca, rp.c)
   # if "kij" not in rp.data:
   # add_xbin_to_respairdat(rp, xbin, kw["min_ssep"])
   if "rotid" not in rp.data:
      add_rots_to_respairdat(rp, rotspace)
   pair_score, order, binkey, binrange = create_res_pair_score_map(rp, xbin, **kw)
   # pair_score.dump("/home/sheffler/debug/rpxdock/datafiles/pair_score.bin")
   pair_range = PHMap_u8u8()
   pair_range[binkey] = binrange
   res1 = np.concatenate([rp.p_resi.data, rp.p_resj.data])[order]
   res2 = np.concatenate([rp.p_resj.data, rp.p_resi.data])[order]
   rps = ResPairScore(xbin, binkey, pair_score, pair_range, res1, res2, rotspace, rp)
   return rps
