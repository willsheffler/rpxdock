import os, time, logging
import numpy as np

log = logging.getLogger(__name__)

class ResPairData:
   def __init__(self, data, sanity_check=None):
      import xarray as xr
      if isinstance(data, xr.Dataset): self.data = data
      elif isinstance(data, ResPairData): self.data = data.data
      if hasattr(self, "data"):
         if sanity_check is True: self.sanity_check()
         return
      else:
         _rp_from_raw_dicts(self, data, sanity_check)

   def __getattr__(self, k):
      if k == "data":
         raise AttributeError
      return getattr(self.data, k)

   def __getitem__(self, k):
      return self.data[k]

   def __str__(self):
      return "ResPairData with data = " + str(self.data).replace("\n", "\n  ")

   def _get_keepers_by_pdb(self, keep, random=True, seed=None, **kw):
      if isinstance(keep, (int, np.int32, np.int64)):
         if random:
            if seed is not None:
               np.random.seed(seed)
            keep = np.random.choice(len(self.pdb), keep, replace=False)
         else:
            keep = np.arange(keep)
      if isinstance(keep, np.ndarray) and keep.dtype == np.bool:
         keep = np.where(keep)[0]
      return np.array(sorted(keep))

   def subset_by_pdb(self, keep, sanity_check=False, update_p_res=True, **kw):
      """keep subset of data in same order as original"""
      keepers = self._get_keepers_by_pdb(keep, **kw)
      if np.sum(keepers) == 0: raise ValueError('no pdbs remain')
      residx = np.isin(self.data.r_pdbid, keepers)
      pairidx = np.isin(self.data.p_pdbid, keepers)

      rpsub = self.data.sel(pdbid=keepers, resid=residx, pairid=pairidx)

      _update_relational_data(rpsub, self.data.pdb_res_offsets, keepers, update_p_res)

      new = ResPairData(rpsub)
      if sanity_check:
         # try:
         new.sanity_check()
      # except AssertionError:
      # import _pickle
      # with open('subset_by_pdb_error')
      return new

   def subset_by_aa(self, aas, sanity_check=False, return_keepers=False):
      if isinstance(aas, str): aas = tuple(aas)
      keepers = np.isin(self.id2aa[self.aaid].data, aas)
      to_return = self.subset_by_res(keepers, sanity_check)
      if return_keepers: to_return = to_return, keepers
      return to_return

   def subset_by_ss(self, ss, sanity_check=False, return_keepers=False):
      if len(set(ss) - set("EHL")) > 0:
         raise ValueError('ss must be EHL')
      if isinstance(ss, str): ss = tuple(ss)
      keepers = np.isin(self.id2ss[self.ssid].data, ss)
      to_return = self.subset_by_res(keepers, sanity_check)
      if return_keepers: to_return = to_return, keepers
      return to_return

   def subset_by_res(self, keepers, sanity_check=False):
      assert len(keepers) == len(self.data.resid)
      if np.sum(keepers) == 0: raise ValueError('no residues remain')

      # t1 = time.perf_counter()
      # remove residues w/o removing pdbs
      rp_res = ResPairData(self.data.sel(resid=keepers))
      g_nres = rp_res.r_pdbid.groupby(rp_res.r_pdbid).count()
      nres = np.zeros(len(self.nres), dtype=np.int32)
      nres[g_nres.r_pdbid.values] = g_nres.values  # sketchy?
      new_pdb_res_offsets = np.concatenate([[0], np.cumsum(nres)])
      rp_res.data.attrs["pdb_res_offsets"] = new_pdb_res_offsets
      rp_res.data.nres.values = nres
      # print('t1', time.perf_counter() - t1)

      # t2 = time.perf_counter()
      # remove pairs
      keep_idx = np.where(keepers)[0]
      p_resi_ok = np.isin(rp_res.p_resi, keep_idx)
      p_resj_ok = np.isin(rp_res.p_resj, keep_idx)
      pair_keep = np.logical_and(p_resi_ok, p_resj_ok)
      if np.sum(pair_keep) == 0: raise ValueError('no pairs left')
      rp_pair = ResPairData(rp_res.data.sel(pairid=pair_keep))
      # print('t2', time.perf_counter() - t2)

      # t3 = time.perf_counter()
      # adjust pair res numbering
      noldi = np.max(rp_pair.data.p_resi).values + 1
      noldj = np.max(rp_pair.data.p_resj).values + 1
      nikeep = np.max(keep_idx) + 1
      old2new = np.zeros(max(nikeep, noldi, noldj), "i4") - 1
      old2new[keep_idx] = np.arange(len(keep_idx))
      rp_pair.data.p_resi.values = old2new[rp_pair.data.p_resi.values]
      rp_pair.data.p_resj.values = old2new[rp_pair.data.p_resj.values]
      assert np.all(rp_pair.p_resi >= 0)
      assert np.all(rp_pair.p_resj >= 0)
      # print('t3', time.perf_counter() - t3)

      # t4 = time.perf_counter()
      # remove empty pdbs and adjust r/p_pdbid
      keep_pdb = np.where(rp_pair.nres > 0)[0]
      old2new = np.zeros(len(rp_pair.nres), 'i4') - 1
      # print(keep_pdb)
      old2new[keep_pdb] = np.arange(len(keep_pdb))
      rp_pair.data = rp_pair.data.sel(pdbid=keep_pdb)
      rp_pair.r_pdbid.values = old2new[rp_pair.r_pdbid.values]
      rp_pair.p_pdbid.values = old2new[rp_pair.p_pdbid.values]
      # print('t4', time.perf_counter() - t4)

      # t5 = time.perf_counter()
      # recompute offsets
      new_pdb_res_offsets = np.concatenate([[0], np.cumsum(rp_pair.nres)])
      rp_pair.data.attrs["pdb_res_offsets"] = new_pdb_res_offsets
      g_npair = rp_pair.p_pdbid.groupby(rp_pair.p_pdbid).count()
      npair = np.zeros(len(rp_pair.nres), dtype=np.int32)
      npair[g_npair.p_pdbid.values] = g_npair.values  # sketchy?
      new_pdb_pair_offsets = np.concatenate([[0], np.cumsum(npair)])
      rp_pair.data.attrs["pdb_pair_offsets"] = new_pdb_pair_offsets
      # print('t5', time.perf_counter() - t5)

      # t6 = time.perf_counter()
      if sanity_check: rp_pair.sanity_check()
      # print('t6', time.perf_counter() - t6)
      return rp_pair

   def subset_by_pair(self, keepers, sanity_check=False, update_p_res=True):
      rpsub = ResPairData(self.data.sel(pairid=keepers))
      pdb_keep = np.unique(rpsub.p_pdbid)
      if len(pdb_keep) == 0: raise ValueError('no pairs remain')
      if len(pdb_keep) != len(rpsub.pdbid):
         # some whole pdbs removed, full update
         rpsub = rpsub.subset_by_pdb(pdb_keep, update_p_res=update_p_res)
      else:
         # no whole pdbs removed, simple update
         tmp = np.cumsum(rpsub.p_pdbid.groupby(rpsub.p_pdbid).count())
         new_pdb_pair_offsets = np.concatenate([[0], tmp])
         rpsub.data.attrs["pdb_pair_offsets"] = new_pdb_pair_offsets

      if sanity_check:
         rpsub.sanity_check()

      return rpsub

   def split_by_pdb(self, frac, random=True, **kw):
      n1 = int(len(self.pdb) * frac)
      n2 = len(self.pdb) - n1
      if random:
         part1 = np.random.choice(len(self.pdb), n1, replace=False)
         part2 = np.array(list(set(range(len(self.pdb))) - set(part1)))
         np.random.shuffle(part2)
      else:
         part1 = np.arange(n1)
         part2 = np.arange(n1, len(self.pdb))
      parts = [sorted(part1), sorted(part2)]
      return [self.subset_by_pdb(part, **kw) for part in parts]

   def sanity_check(self):
      rp = self.data
      Npdb = len(self.pdb)
      # from rpxdock import Timer
      # with Timer() as timer:
      if True:
         log.debug(f'ResPaisDat sanity_check iters {min(Npdb, 100)}')
         for ipdb in np.random.choice(Npdb, min(Npdb, 100), replace=False):
            rlb, rub = rp.pdb_res_offsets[ipdb:ipdb + 2]
            # if rlb > 0:
            # assert rp.r_pdbid[rlb - 1] == ipdb - 1
            assert rp.r_pdbid[rlb] == ipdb
            assert rp.r_pdbid[rub - 1] == ipdb
            if ipdb + 1 < len(rp.pdb):
               assert rp.r_pdbid[rub] == ipdb + 1
            p_resno = rp.resno[rlb:rub]
            assert np.all(p_resno >= rp.resid[rlb:rub] - rlb)
            # timer.checkpoint('check pdb_res_offsets')

            plb = rp.pdb_pair_offsets[ipdb]
            pub = rp.pdb_pair_offsets[ipdb + 1]
            if plb == pub: continue
            p_resi = rp.p_resi[plb:pub] - rp.pdb_res_offsets[ipdb]
            p_resj = rp.p_resj[plb:pub] - rp.pdb_res_offsets[ipdb]
            if np.min(p_resi) < 0 or np.max(p_resi) >= rp.nres[ipdb]:
               r = str(np.random.randint(1e9))
               # import _pickle
               #
               # with open(r + ".pickle", "wb") as out:
               # _pickle.dump(rp, out)
               print(r, "sanity_check fail")
               print(r, "pdb", ipdb, rp.pdb[ipdb].values, rp.nres[ipdb].values)
               print(r, "offset res", rp.pdb_res_offsets[ipdb])
               print(r, "pair_range", plb, pub)
               print(r, p_resi)
               print(r, p_resj)
               # import sys
               #
               # sys.exit()

            assert np.min(p_resi) >= 0
            assert np.max(p_resi) < rp.nres[ipdb]
            assert 0 < np.min(p_resj) < rp.nres[ipdb]
            # timer.checkpoint('check pdb_pair_offsets vs nres')

            resi_this_ipdb = rp.p_resi[rp.p_pdbid == ipdb]
            assert np.all(rp.r_pdbid[resi_this_ipdb] == ipdb)
            # timer.checkpoint('loop iter end')

         # timer.checkpoint('big loop')
         # sanity check pair distances vs res-res distances
         if 'cb' in rp:
            cbi = rp.cb[rp.p_resi]
            cbj = rp.cb[rp.p_resj]
            dhat = np.linalg.norm(cbi - cbj, axis=1)
            assert np.allclose(dhat, rp.p_dist, atol=1e-3)
         # timer.checkpoint('coords/dist check')

         assert 0 <= np.max(rp.p_resi) < len(rp.resid)
         assert 0 <= np.max(rp.p_resj) < len(rp.resid)
         # timer.checkpoint('p_res bounds check')
      # print(timer)

   def only_whats_needed(self, task):
      raise NotImplementedError('why is this being called?')
      not_needed = dict(
         seqproftest=
         "phi psi omega chi1 chi2 chi3 chi4 chain r_fa_sol r_fa_intra_atr_xover4 r_fa_intra_rep_xover4 r_fa_intra_sol_xover4 r_lk_ball r_lk_ball_iso r_lk_ball_bridge r_lk_ball_bridge_uncpl r_fa_elec r_fa_intra_elec r_pro_close r_hbond_sr_bb r_hbond_lr_bb r_hbond_bb_sc r_hb_sc r_dslf_fa13 r_rama_prepro r_omega r_p_aa_pp r_fa_dun_rot r_fa_dun_dev r_fa_dun_semi r_hxl_tors r_ref sasa2 sasa4 nnb6 nnb8 nnb12 nnb14 p_hb_bb_bb p_hb_bb_sc p_hb_sc_bb p_hb_sc_sc p_fa_atr p_fa_rep p_fa_sol p_lk_ball p_fa_elec p_hbond_sr_bb p_hbond_lr_bb"
         .split(),
         respairscore=
         "r_fa_intra_atr_xover4 r_fa_intra_rep_xover4 r_fa_intra_sol_xover4 r_lk_ball_iso r_lk_ball_bridge r_lk_ball_bridge_uncpl r_fa_intra_elec r_pro_close r_rama_prepro r_omega r_p_aa_pp r_hxl_tors r_ref sasa2 sasa4 nnb6 nnb8 nnb12 nnb14 p_hb_bb_bb p_hb_bb_sc p_hb_sc_bb p_fa_atr p_fa_rep p_fa_sol p_lk_ball p_fa_elec"
         .split(),
      )
      assert task in not_needed
      for v in not_needed[task]:
         if not v in self.data:
            print("ResPairData.only_whats_needed: missing:", v)
      return ResPairData(self.data.drop(not_needed[task]))

def _get_pdb_names(files):
   base = [os.path.basename(f) for f in files]
   assert all(b[4:] == "_0001.pdb" for b in base)
   return [b[:4] for b in base]

def _update_relational_data(data, prev_pdb_res_offsets, pdb_subset=None, update_p_res=True):
   # update relational stuff
   if pdb_subset is None:
      pdb_subset = np.arange(len(data.pdbid))

   # update pdbid references
   nold = np.max(data.r_pdbid).values + 1
   old2new = np.zeros(nold, "i4") - 1
   old2new[pdb_subset] = np.arange(len(pdb_subset))

   data.r_pdbid.values = old2new[data.r_pdbid.values]
   data.p_pdbid.values = old2new[data.p_pdbid.values]
   new_pdb_res_offsets = np.concatenate([[0], np.cumsum(data.nres)])
   data.attrs["pdb_res_offsets"] = new_pdb_res_offsets

   tmp = np.cumsum(data.p_pdbid.groupby(data.p_pdbid).count())
   new_pdb_pair_offsets = np.concatenate([[0], tmp])
   data.attrs["pdb_pair_offsets"] = new_pdb_pair_offsets
   old_res_ofst = prev_pdb_res_offsets[pdb_subset[data.p_pdbid.values]]
   new_res_ofst = new_pdb_res_offsets[data.p_pdbid.values]
   if update_p_res:
      data["p_resi"] += new_res_ofst - old_res_ofst
      data["p_resj"] += new_res_ofst - old_res_ofst
   assert np.all(data.p_resi >= 0)
   assert np.all(data.p_resj >= 0)

def _change_seq_ss_to_ids(rp):
   import xarray as xr
   dat = rp.data
   id2aa = np.array(list("ACDEFGHIKLMNPQRSTVWY"))
   aa2id = xr.DataArray(np.arange(20, dtype="i4"), [("aa", id2aa)])
   aaid = aa2id.sel(aa=dat.seq).values.astype("i4")
   dat["aaid"] = xr.DataArray(aaid, dims=["resid"])
   if hasattr(dat, 'drop_vars'): dat = dat.drop_vars("seq")
   else: dat = dat.drop("seq")
   dat["id2aa"] = xr.DataArray(id2aa, [aa2id], ["aai"])
   dat["aa2id"] = xr.DataArray(aa2id, [id2aa], ["aa"])

   id2ss = np.array(list("EHL"))
   ss2id = xr.DataArray(np.arange(3, dtype="i4"), [("ss", id2ss)])
   ssid = ss2id.sel(ss=dat.ss).values.astype("i4")
   dat["ssid"] = xr.DataArray(ssid, dims=["resid"])
   if hasattr(dat, 'drop_vars'): dat = dat.drop_vars("ss")
   else: dat = dat.drop("ss")
   dat["id2ss"] = xr.DataArray(id2ss, [ss2id], ["ssi"])
   dat["ss2id"] = xr.DataArray(ss2id, [id2ss], ["ss"])
   rp.data = dat

def _rp_from_raw_dicts(self, data, sanity_check):
   # loading from raw dicts
   import xarray as xr
   assert isinstance(data, dict)

   raw = data
   pdbdata = raw["pdbdata"]
   coords = raw["coords"]
   resdata = raw["resdata"]
   pairdata = raw["pairdata"]
   pdb_res_offsets = raw["pdb_res_offsets"]
   pdb_pair_offsets = raw["pdb_pair_offsets"]
   bin_params = raw["bin_params"]

   # put this stuff in dataset

   self.res_ofst = pdb_res_offsets
   self.pair_ofst = pdb_pair_offsets
   self.bin_params = bin_params

   pdbdata["file"] = pdbdata["pdb"]
   pdbdata["pdb"] = _get_pdb_names(pdbdata["file"])
   for k, v in pdbdata.items():
      pdbdata[k] = (["pdbid"], v)
   pdbdata["com"] = (["pdbid", "xyzw"], pdbdata["com"][1])

   for k, v in resdata.items():
      resdata[k] = (["resid"], v)
   # resdata["n"] = (["resid", "xyzw"], coords["ncac"][:, 0])
   # resdata["ca"] = (["resid", "xyzw"], coords["ncac"][:, 1])
   # resdata["c"] = (["resid", "xyzw"], coords["ncac"][:, 2])
   resdata["n"] = (["resid", "xyzw"], coords["n"])
   resdata["ca"] = (["resid", "xyzw"], coords["ca"])
   resdata["c"] = (["resid", "xyzw"], coords["c"])
   resdata["o"] = (["resid", "xyzw"], coords["o"])
   resdata["cb"] = (["resid", "xyzw"], coords["cb"])
   resdata["stub"] = (["resid", "hrow", "hcol"], coords["stubs"])
   resdata["r_pdbid"] = resdata["pdbno"]
   del resdata["pdbno"]

   for k, v in pairdata.items():
      pairdata[k] = (["pairid"], v)
   pairdata["p_pdbid"] = pairdata["pdbno"]
   del pairdata["pdbno"]

   if len(pdbdata.keys() & resdata.keys()):
      print(pdbdata.keys() & resdata.keys())
      assert 0 == len(pdbdata.keys() & resdata.keys())
   if len(pdbdata.keys() & pairdata.keys()):
      print(pdbdata.keys() & pairdata.keys())
      assert 0 == len(pdbdata.keys() & pairdata.keys())
   if len(pairdata.keys() & resdata.keys()):
      print(pairdata.keys() & resdata.keys())
      assert 0 == len(pairdata.keys() & resdata.keys())

   data = {**pdbdata, **resdata, **pairdata}
   assert len(data) == len(pdbdata) + len(resdata) + len(pairdata)
   self.data = xr.Dataset(
      data,
      coords=dict(
         xyzw=["x", "y", "z", "w"],
         hrow=["x", "y", "z", "w"],
         hcol=["x", "y", "z", "t"],
      ),
      attrs=dict(
         pdb_res_offsets=pdb_res_offsets,
         pdb_pair_offsets=pdb_pair_offsets,
         xbin_params=bin_params,
         xbin_types=raw["xbin_types"],
         xbin_swap_type=raw["xbin_swap_type"],
         eweights=raw["eweights"],
      ),
   )

   _change_seq_ss_to_ids(self)

   res_ofst = self.pdb_res_offsets[self.p_pdbid]
   self.data["p_resi"] += res_ofst
   self.data["p_resj"] += res_ofst

   assert self.data.stub.sel(hcol="t").shape[1] == 4
   assert np.all(self.data.ca.sel(xyzw="w") == 1)
   assert np.all(self.data.cb.sel(xyzw="w") == 1)

   if sanity_check is not False:
      self.sanity_check()
