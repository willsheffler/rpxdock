import willutil as wu
import rpxdock as rp
import numpy as np

def main():
   test_inline_onecomp()

def slide_dock_oligomer(sym, psym, nsym, xyz, step=1.0, clash_radius=2.4, contact_dis=8):
   nrot = int(360 // int(psym[1]) // step)
   step = np.radians(step)
   startaxis = wu.hnormalized(xyz.mean(axis=(0, 1, 2)))
   paxis = wu.sym.axes(sym=sym, nfold=psym)
   naxis = wu.sym.axes(sym=sym, nfold=nsym)

   ic(startaxis, paxis, naxis)
   initalign = wu.halign(startaxis, paxis)
   tonbr = wu.hrot(naxis, nfold=int(nsym[1]))
   paxis2 = wu.hxform(tonbr, paxis)
   ic(tonbr)
   xyz = wu.hxform(initalign, xyz)
   axisang = wu.hangle(paxis, naxis)
   slidedirn = wu.hnormalized(wu.hxform(tonbr, paxis) - paxis)
   bvh = wu.cpp.bvh.BVH(xyz[:, :, :].reshape(-1, 3))

   best = 0, None, None
   for i in range(nrot):
      pos1 = wu.hrot(paxis, i * step)
      pos2 = tonbr @ pos1
      delta = wu.cpp.bvh.bvh_slide(bvh, bvh, pos1, pos2, clash_radius, slidedirn[:3])
      delta = delta / np.sin(axisang)
      # ic(delta)
      # assert 0
      # t.checkpoint('slide')
      pos1 = wu.htrans(-paxis * delta / 2, doto=pos1)
      pos2 = wu.htrans(-paxis2 * delta / 2, doto=pos2)
      score = wu.cpp.bvh.bvh_count_pairs_vec(bvh, bvh, pos1, pos2, contact_dis)
      if score > best[0]:
         best = score, pos1, pos2
         ic(i, best[0])
         # wu.showme(np.concatenate([wu.hxform(pos1, xyz), wu.hxform(pos2, xyz)]), name=f'dock{i}')
      # assert 0
   ic(best)
   # score, pos1, pos2 = best
   # wu.showme(np.concatenate([wu.hxform(pos1, xyz), wu.hxform(pos2, xyz)]), name=f'dock{i}')
   newxyz = wu.hxform(pos1, xyz[0])

   # find closest to orig com
   com = wu.hcom(xyz[0, :, 1])
   symcom = wu.hxform(wu.sym.frames(sym), wu.hcom(newxyz[:, 1]))
   f = wu.sym.frames(sym)[np.argmin(wu.hnorm(com - symcom))]
   newxyz = wu.hxform(f, newxyz)

   return newxyz

def test_inline_onecomp():
   pdb = wu.readpdb('/home/sheffler/project/symmmotif_HE/input/test_trimer_icos_2.pdb')
   xyz = pdb.ncac(splitchains=True)
   np.save('/tmp/xyz.npy', xyz)
   # xyz = np.load('/tmp/xyz.npy')

   print(xyz.shape)
   # wu.showme(xyz)
   ic(wu.hcom(xyz.reshape(-1, 3)))

   rpxscore = wu.load(
      # '/home/sheffler/data/rpx/hscore/willsheffler/ilv_h/pdb_res_pair_data_si30_rots_H_ILV_SSindep_p0.5_b1_hier4_Kflat_1_0.pickle',
      '/home/sheffler/data/rpx/hscore/willsheffler/ilv_h/pdb_res_pair_data_si30_rots_H_ILV_SSindep_p0.5_b1_hier0_Kflat_2_1.pickle'
      # '/home/sheffler/data/rpx/hscore/willsheffler/ilv_h/100/pdb_res_pair_data_si30_100_rots_H_ILV_SSindep_p0.5_b1_hier4_Kflat_1_0.pickle.bz2.xmap.txz.pickle'
   )

   with wu.Timer():
      newxyz = slide_dock_oligomer('icos', 'c3', 'c2', xyz)
   wu.dumppdb('test.pdb', wu.hxform(wu.sym.frames('icos'), newxyz))

   # stub = rp.motif.bb_stubs(xyz.reshape(-1, 3, 3))
   # ic(stub.shape)
   # pairs, lbub = rp.bvh.bvh_collect_pairs_range_vec(bvh, bvh, pos1, pos2, 8)
   # t.checkpoint('pairs')
   # pscore = rp.xbin.xbin_util.map_pairs_multipos(rpxscore.xbin, rpxscore.phmap, pairs, stub, stub, lbub, pos1,
   # pos2, incomplete_ok=True)
   # lbub1, lbub2, idx1, idx2, ressc1, ressc2 = rp.motif.marginal_max_score(lbub, pairs, pscore)
   # score = (ressc1.mean() + ressc2.mean()) / 2
   # ic(score)

   # xyz = wu.readpdb('/home/sheffler/project/symmmotif_HE/input/1coi_x_trim2.pdb').ncac(splitchains=True)
   # bvh1 =
   # ic(xyz.shape)
   # pscore = rp.xbin.xbin_util.map_pairs_multipos(
   # rpxscore.xbin,
   # rpxscore.phmap,
   # pairs,
   # stub,
   # stub,
   # lbub,
   # pos1,
   # pos2,
   # incomplete_ok=True,
   # )
   # assert 0

if __name__ == '__main__':
   main()
