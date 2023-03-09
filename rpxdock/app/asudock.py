#!/home/sheffler/bin/python_rpxdock

import sys, pytest, os

sys.path.append('/home/sheffler/src/rpxdock_master')
import numpy as np
import rpxdock as rp

import willutil as wu
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor

def main():
   kw = rp.options.get_cli_args()
   asymdock_pdbs(**kw)
   ic('test_asym.py DONE')

# def asudock

def asymdock_pdbs(**kw):
   kw = wu.Bunch(kw)
   # hscore = rp.CachedProxy(rp.RpxHier(kw.hscore_files, **kw))
   hscore = rp.RpxHier('ilv_h/1000', hscore_data_dir='/home/sheffler/data/rpx/hscore/willsheffler')
   # kw = _test_args()
   sym = kw.architecture
   kw.output_prefix = 'diffuse_'
   kw.beam_size = 5e4
   kw.clashdis = 3.0
   limit_rotation = 0.4
   # limit_rotation = np.pi / 3
   cartlb = np.array([-20, -20, -20])
   cartub = np.array([+20, +20, +20])
   cartbs = np.array([5, 5, 5], dtype='i')
   sampler = rp.sampling.XformHier_f4(cartlb, cartub, cartbs, 20)
   kw.wts = wu.Bunch(ncontact=0.1, rpx=1.0)
   kw.max_bb_redundancy = 0.666

   # ax2 = wu.sym.axes(sym)[2]
   # ax3 = wu.sym.axes(sym)[3]
   # frames = [np.eye(4), wu.hrot(ax2, 180), wu.hrot(ax3, 120), wu.hrot(ax3, 240)]

   userosetta = True

   for pdb in kw.inputs1:
      if not kw.skip_errors:
         asudock_pdb(sym, pdb, userosetta, hscore, sampler, limit_rotation, **kw)
      else:
         try:
            asudock_pdb(sym, pdb, userosetta, hscore, sampler, limit_rotation, **kw)
         except (ValueError, AssertionError) as e:
            ic(e)
            ic('ERROR fail on', pdb)
            # raise e
            continue

def asudock_pdb(sym, pdb, userosetta, hscore, sampler, limit_rotation, **kw):
   kw = wu.Bunch(kw)
   userosetta = True
   # assert not 'pyrosetta' in sys.modules
   kw.ignored_aas = 'CP'
   body0 = rp.Body(pdb, extract_chain=0, userosetta=userosetta, **kw)
   # assert not 'pyrosetta' in sys.modules

   assert userosetta or 'pyrosetta' not in sys.modules

   # for EXPAND in (1, 1.1, 1.2):
   EXPAND = 1.1
   if True:
      # body.pos ?????
      xdelta = wu.htrans((EXPAND - 1) * body0.com())
      body = body0.copy_xformed(xdelta, **kw)
      ic(len(body.bvh_cen), len(body0.bvh_cen))

      if body.crystinfo is not None:
         ci = body.crystinfo
         assert sym.split('_')[0].replace(' ', '') == ci.spacegroup.replace(' ', '')
         assert max(ci.cell) == min(ci.cell)
         cellsize = ci.cell[0] * EXPAND
      ic(cellsize)
      frames = wu.sym.frames(sym, cells=3, cellsize=cellsize, center=body.com(), asucen=body.com(), xtalrad=0.5)
      primaryframes = wu.sym.frames(sym, cells=None, cellsize=cellsize)

#      for i, f in enumerate(frames):
#         body.pos = f
#         body.dump_pdb(f'frame_{EXPAND}_{i:03}.pdb')
#   assert 0

   preserve_contacts = False
   bodies0 = [body.copy() for b in frames[:-1]]
   bodies1 = [body.copy() for b in frames[1:]]
   if preserve_contacts:
      rsets0, rsets1 = set_allowed_residues_by_current_neighbors(body=body, frames=primaryframes, **kw)
      for i in range(len(primaryframes) - 1):
         bodies0[i].set_allowed_residues(required_res_sets=[rsets0[i]])
         bodies1[i].set_allowed_residues(required_res_sets=[rsets1[i]])

   # body = body.copy_xformed(wu.htrans((ax2 + ax2) * 40))
   x2asymcen = wu.htrans(-body.com())
   # body = body.copy_xformed(wu.htrans(-body.com()))

   result = rp.search.make_asym(
      [bodies0, bodies1],
      hscore,
      sampler,
      frames=frames,
      x2asymcen=x2asymcen,
      limit_rotation=limit_rotation,
      sym=sym,
      **kw,
   )
   scores = list()
   for iframe in range(1, len(frames)):
      scores.append(
         hscore.scorepos(
            body,
            body,
            frames[0] @ result.xforms.data[0],
            frames[iframe] @ result.xforms.data[0],
            4,
            **kw,
         ))
   minscore = min(scores[0], np.mean(scores[1:3]))

   if minscore < 2.0:
      raise ValueError(f'score to low {scores}')
   print(result)
   pre = kw.output_prefix + os.path.basename(pdb).replace('.pdb', '').replace('.gz', '') + '_rp1'
   # body.dump_pdb(f'{pre}_init.pdb', symframes=frames)
   result.dump_pdbs_top_score(kw.nout_top, symframes=frames, output_prefix=pre, dump_input=True)

   # # rp.search.result_to_tarball(result, 'rpxdock/data/testdata/test_asym.result', overwrite=True)
   # ref = rp.data.get_test_data('test_asym')
   # try:
   #    rp.search.assert_results_close(result, ref)
   # except AssertionError:
   #    print('WARNING full results for asym docking dont match... checking scores only')
   #    assert np.allclose(ref.scores, result.scores, atol=1e-6)

def set_allowed_residues_by_current_neighbors(body, frames, **kw):
   # calc allowed_res1 and 2
   rsets0, rsets1 = list(), list()

   # body.dump_pdb('test.pdb', symframes=frames)
   # assert 0

   for iframe, frame in enumerate(frames):
      if iframe in [0]: continue
      for maxdis in [d2 / 2 for d2 in range(5, 40)]:
         pairs = body.contact_pairs(body, frames[0], frame, maxdis=maxdis)
         required_res_set0 = set(pairs[:, 0])
         required_res_set1 = set(pairs[:, 1])
         # ic(iframe, pairs)
         # for r in rsets0:
         # required_res_set0 -= r
         if len(required_res_set0) >= 7 and len(required_res_set1) >= 7:
            # ic('break', iframe, maxdis, len(required_res_set0), len(required_res_set1))
            rsets0.append(required_res_set0)
            rsets1.append(required_res_set1)
            break
      else:
         body.pos = frames[0]
         body.dump_pdb('frame0.pdb')
         body.pos = frames[iframe]
         body.dump_pdb(f'frame{iframe}.pdb')
         assert 0, f'no neighbors found for frame {iframe} maxdis {maxdis}'
   rsets0 = [list(sorted(r)) for r in rsets0]
   rsets1 = [list(sorted(r)) for r in rsets1]
   return rsets0, rsets1

def _test_args():
   kw = rp.app.defaults()
   kw.wts = wu.Bunch(ncontact=0.01, rpx=1.0)
   kw.beam_size = 1e4
   kw.max_bb_redundancy = 3.0
   kw.max_longaxis_dot_z = 0.5
   if not 'pytest' in sys.modules:
      kw.executor = ThreadPoolExecutor(min(4, kw.ncpu / 2))
   kw.multi_iface_summary = np.min
   kw.debug = True
   return kw

if __name__ == '__main__':
   main()
'''
   ic| 'rosetta', self.coord.shape: (100, 5, 4)
ic| required_res_sets: None
ic| self.coord.shape: (100, 5, 4)
ic| self.seq: array(['S', 'E', 'E', 'E', 'L', 'L', 'E', 'L', 'L', 'A', 'K', 'E', 'L',
                     'A', 'L', 'A', 'A', 'L', 'L', 'A', 'L', 'L', 'A', 'L', 'L', 'L',
                     'L', 'L', 'E', 'K', 'L', 'S', 'K', 'E', 'E', 'L', 'L', 'K', 'A',
                     'L', 'L', 'E', 'L', 'L', 'E', 'A', 'A', 'A', 'K', 'L', 'L', 'G',
                     'S', 'E', 'E', 'A', 'E', 'L', 'L', 'A', 'L', 'L', 'L', 'K', 'L',
                     'L', 'L', 'G', 'N', 'E', 'E', 'K', 'A', 'K', 'K', 'L', 'L', 'K',
                     'L', 'L', 'L', 'K', 'L', 'L', 'S', 'P', 'E', 'A', 'L', 'A', 'E',
                     'L', 'L', 'K', 'A', 'L', 'L', 'L', 'L', 'L'], dtype='<U1')
ic| len(self.seq): 100
ic| r: 'Times(name=Timer, order=longest, summary=sum):
            total    5.38185'
ic| required_res_sets: [[4, 7, 56, 60, 64, 89, 93, 96, 99]]
ic| required_res_sets: [[4, 7, 56, 60, 64, 89, 93, 96, 99]]
ic| required_res_sets: [[8, 12, 15, 16, 19, 20, 22, 23, 26, 27, 83]]
ic| required_res_sets: [[9, 16, 17, 20, 38, 41, 42, 45, 46, 49]]
ic| required_res_sets: [[9, 16, 17, 20, 38, 41, 42, 45, 46, 49]]
ic| required_res_sets: [[8, 12, 15, 16, 19, 20, 22, 23, 26, 27, 83]]
ic| kw.ibody: 0, i: 0, ncontains: [1, 1]
ic| kw.ibody: 0, i: 1, ncontains: [2, 2]
ic| kw.ibody: 0, i: 2, ncontains: [0, 0]
ic| kw.ibody: 1, i: 0, ncontains: [0, 0]
ic| kw.ibody: 1, i: 1, ncontains: [0, 0]
ic| kw.ibody: 1, i: 2, ncontains: [0, 0]
ic| kw.ibody: 2, i: 0, ncontains: [26, 24]
ic| kw.ibody: 2, i: 1, ncontains: [30, 21]
ic| kw.ibody: 2, i: 2, ncontains: [18, 20]
INFO:rpxdock.search.hierarchical:rpxdock iresl 0 ntot   1,179,648 nonzero   101
ic| kw.ibody: 0, i: 0, ncontains: [1, 1]
ic| kw.ibody: 0, i: 1, ncontains: [3, 3]
ic| kw.ibody: 0, i: 2, ncontains: [4, 4]
ic| kw.ibody: 1, i: 0, ncontains: [23, 13]
ic| kw.ibody: 1, i: 1, ncontains: [15, 11]
ic| kw.ibody: 1, i: 2, ncontains: [20, 8]
ic| kw.ibody: 2, i: 0, ncontains: [13, 23]
ic| kw.ibody: 2, i: 1, ncontains: [11, 15]
ic| kw.ibody: 2, i: 2, ncontains: [21, 26]
INFO:rpxdock.search.hierarchical:rpxdock iresl 1 ntot       7,744 nonzero   469
ic| kw.ibody: 0, i: 0, ncontains: [14, 14]
ic| kw.ibody: 0, i: 1, ncontains: [14, 14]
ic| kw.ibody: 0, i: 2, ncontains: [15, 15]
ic| kw.ibody: 1, i: 0, ncontains: [11, 5]
ic| kw.ibody: 1, i: 1, ncontains: [9, 5]
ic| kw.ibody: 1, i: 2, ncontains: [10, 5]
ic| kw.ibody: 2, i: 0, ncontains: [16, 18]
ic| kw.ibody: 2, i: 1, ncontains: [14, 18]
ic| kw.ibody: 2, i: 2, ncontains: [18, 31]
INFO:rpxdock.search.hierarchical:rpxdock iresl 2 ntot      30,016 nonzero 1,860
ic| kw.ibody: 0, i: 0, ncontains: [13, 13]
ic| kw.ibody: 0, i: 1, ncontains: [16, 16]
ic| kw.ibody: 0, i: 2, ncontains: [16, 16]
ic| kw.ibody: 1, i: 0, ncontains: [34, 29]
ic| kw.ibody: 1, i: 1, ncontains: [33, 29]
ic| kw.ibody: 1, i: 2, ncontains: [35, 29]
ic| kw.ibody: 2, i: 0, ncontains: [29, 34]
ic| kw.ibody: 2, i: 1, ncontains: [29, 33]
ic| kw.ibody: 2, i: 2, ncontains: [29, 35]
INFO:rpxdock.search.hierarchical:rpxdock iresl 3 ntot      40,000 nonzero 2,617
ic| kw.ibody: 0, i: 0, ncontains: [14, 14]
ic| kw.ibody: 0, i: 1, ncontains: [14, 14]
ic| kw.ibody: 0, i: 2, ncontains: [14, 14]
ic| kw.ibody: 1, i: 0, ncontains: [35, 19]
ic| kw.ibody: 1, i: 1, ncontains: [42, 26]
ic| kw.ibody: 1, i: 2, ncontains: [42, 25]
ic| kw.ibody: 2, i: 0, ncontains: [19, 35]
ic| kw.ibody: 2, i: 1, ncontains: [25, 42]
ic| kw.ibody: 2, i: 2, ncontains: [20, 34]
INFO:rpxdock.search.hierarchical:rpxdock iresl 4 ntot      40,000 nonzero 2,881
'''