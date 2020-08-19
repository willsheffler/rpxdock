import os, numpy as np, rpxdock as rp, rmsd

def _reduce_dataset(dat):
   return dat.drop([
      't_fa_atr', 't_fa_rep', 't_fa_sol', 't_fa_intra_atr_xover4', 't_fa_intra_rep_xover4',
      't_fa_intra_sol_xover4', 't_lk_ball', 't_lk_ball_iso', 't_lk_ball_bridge',
      't_lk_ball_bridge_uncpl', 't_fa_elec', 't_fa_intra_elec', 't_pro_close', 't_hbond_sr_bb',
      't_hbond_lr_bb', 't_hbond_bb_sc', 't_hb_sc', 't_dslf_fa13', 't_rama_prepro', 't_omega',
      't_p_aa_pp', 't_fa_dun_rot', 't_fa_dun_dev', 't_fa_dun_semi', 't_hxl_tors', 't_ref',
      'r_fa_sol', 'r_fa_intra_atr_xover4', 'r_fa_intra_rep_xover4', 'r_fa_intra_sol_xover4',
      'r_lk_ball', 'r_lk_ball_iso', 'r_lk_ball_bridge', 'r_lk_ball_bridge_uncpl', 'r_fa_elec',
      'r_fa_intra_elec', 'r_pro_close', 'r_hbond_sr_bb', 'r_hbond_lr_bb', 'r_hbond_bb_sc',
      'r_hb_sc', 'r_dslf_fa13', 'r_rama_prepro', 'r_omega', 'r_p_aa_pp', 'r_fa_dun_rot',
      'r_fa_dun_dev', 'r_fa_dun_semi', 'r_hxl_tors', 'r_ref', 'sasa2', 'sasa3', 'sasa4', 'nnb6',
      'nnb8', 'nnb10', 'nnb12', 'nnb14', 'p_raw_etot', 'p_hb_bb_bb', 'p_hb_bb_sc', 'p_hb_sc_bb',
      'p_hb_sc_sc', 'p_fa_sol', 'p_lk_ball', 'p_fa_elec'
   ])

def main():

   min_helix1_len = 14
   max_helix1_len = 28
   min_turn_len = 3
   max_turn_len = 5
   min_helix2_len = 14
   max_helix2_len = 28
   max_e_per_res = -0.3
   max_ca_redundancy = 2.0

   # fname = '/home/sheffler/data/respairdat/pdb_res_pair_data_si30_1000.pickle'
   fname = '/home/sheffler/data/respairdat/pdb_res_pair_data_si30.pickle'
   newfname = fname.replace('/home/sheffler/data/respairdat/pdb_res_pair_data', 'tmp_hlh')

   if not os.path.exists(newfname):
      print('CREATING', newfname)
      rp.dump(_reduce_dataset(rp.load(fname)), newfname)
   print(os.getcwd(), newfname)
   dat = rp.load(newfname)
   dat = rp.ResPairData(dat, sanity_check=True)
   assert np.all(dat.p_resi < dat.p_resj)
   E, H, L = [np.where(dat.ss == aa)[0][0] for aa in 'EHL']
   ss = dat.ssid.data.copy()
   ss[dat.pdb_res_offsets[:-1]] = -1
   # print(ss[:50])

   transitions = ss[:-1] != ss[1:]
   # print(transitions[:50].astype('i'))
   transitions = np.where(transitions)[0]
   tbefore = ss[transitions]
   tafter = ss[transitions + 1]

   print(transitions[:10])
   print(tbefore[:10])
   print(tafter[:10])
   # trans = np.stack([transitions, tbefore, tafter], axis=1)

   hlh = np.logical_and(np.logical_and(tbefore[:-1] == H, tafter[:-1] == L),
                        np.logical_and(tbefore[+1:] == L, tafter[+1:] == H))
   hlh = np.where(hlh)[0]
   # print(hlh)
   hlh_beg = transitions[hlh - 1] + 1
   hlh_end = transitions[hlh + 2] + 1

   print('=============================')
   # for i, j in zip(hlh_beg, hlh_end):
   # print(i, j, ss[i - 1:j + 1])
   # print(i, j, ss[i:j])

   hlh_bounds = np.stack([
      transitions[hlh - 1] + 1,
      transitions[hlh + 0] + 1,
      transitions[hlh + 1] + 1,
      transitions[hlh + 2] + 1,
   ], axis=1)
   # print(hlh_bounds[:10])
   h1len = hlh_bounds[:, 1] - hlh_bounds[:, 0]
   turnlen = hlh_bounds[:, 2] - hlh_bounds[:, 1]
   h2len = hlh_bounds[:, 3] - hlh_bounds[:, 2]
   # ok = np.ones(len(h1len), dtype=np.bool_)
   ok = dat.chain[hlh_bounds[:, 0]] == dat.chain[hlh_bounds[:, 3] - 1]
   ok = np.logical_and(ok, h1len >= min_helix1_len)
   # ok = np.logical_and(ok, h1len <= max_helix1_len)
   ok = np.logical_and(ok, turnlen >= min_turn_len)
   ok = np.logical_and(ok, turnlen <= max_turn_len)
   ok = np.logical_and(ok, h2len >= min_helix2_len)
   # ok = np.logical_and(ok, h2len <= max_helix2_len)
   assert np.any(ok)
   hlh_bounds = hlh_bounds[ok]
   print('HH len cut: keep', len(hlh_bounds), 'of', len(h1len))
   h1len = hlh_bounds[:, 1] - hlh_bounds[:, 0]
   turnlen = hlh_bounds[:, 2] - hlh_bounds[:, 1]
   h2len = hlh_bounds[:, 3] - hlh_bounds[:, 2]

   hlh_bounds[:, 0] = hlh_bounds[:, 1] - np.minimum(h1len, max_helix1_len)
   hlh_bounds[:, 3] = hlh_bounds[:, 2] + np.minimum(h1len, max_helix2_len)

   ok2 = np.zeros(len(hlh_bounds), dtype=np.bool_)
   for idx, (i, j, k, l) in enumerate(hlh_bounds):
      pok = np.ones(len(dat.p_resi), dtype=np.bool_)
      pok = np.logical_and(pok, i <= dat.p_resi.data)
      pok = np.logical_and(pok, j > dat.p_resi.data)
      pok = np.logical_and(pok, k <= dat.p_resj.data)
      pok = np.logical_and(pok, l > dat.p_resj.data)
      if np.sum(pok) == 0: continue
      eres = np.sum(dat.p_etot.data[pok]) / (j - i + l - k)
      if eres > max_e_per_res: continue
      ok2[idx] = True
      # print(eres, ss[i:j], ss[j:k], ss[k:l])
   hlh_bounds = hlh_bounds[ok2]
   print('energy cut: keep', len(hlh_bounds), 'of', len(h1len), 'max_e_per_res =', max_e_per_res)
   h1len = hlh_bounds[:, 1] - hlh_bounds[:, 0]
   turnlen = hlh_bounds[:, 2] - hlh_bounds[:, 1]
   h2len = hlh_bounds[:, 3] - hlh_bounds[:, 2]

   keep = list()
   for tl in range(min_turn_len, max_turn_len + 1):
      sel = turnlen == tl
      clust_lb = hlh_bounds[sel, 1] - min_helix1_len
      clust_ub = hlh_bounds[sel, 2] + max_helix2_len
      idx = np.stack(np.arange(lb, ub) for lb, ub in zip(clust_lb, clust_ub))
      coords = np.stack([dat.n.data[idx], dat.ca.data[idx], dat.c.data[idx]], axis=2)
      coords = coords.reshape(len(idx), -1, 4)
      coords[0] -= rmsd.centroid(coords[0])
      for i in range(1, len(coords)):
         coords[i] -= rmsd.centroid(coords[i])
         coords[i] = np.dot(coords[i], rmsd.kabsch(coords[i], coords[0]))
      # for i in range(len(coords)):
      # rp.io.dump_pdb_from_points('test%04i.pdb' % i, coords[i])

      coords = coords.reshape(len(coords), -1)
      centers = rp.cluster.cookie_cutter(coords, max_ca_redundancy)
      # print(tl, len(coords), len(centers))n
      keep.append(hlh_bounds[sel][centers])

   hlh_bounds_clust = np.concatenate(keep)
   print('clustering: keep', len(hlh_bounds_clust), 'of', len(hlh_bounds), 'max_ca_redundancy =',
         max_ca_redundancy)

def rmsd_aligned_centered(A, B):
   U = rmsd.kabsch(A, B)
   return np.dot(A, U)

if __name__ == '__main__':
   main()
