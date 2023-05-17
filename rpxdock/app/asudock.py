#!/home/sheffler/bin/python_rpxdock

import sys, pytest, os

sys.path.append('/home/sheffler/src/rpxdock_master')
import numpy as np
import rpxdock as rp

import willutil as wu
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor

_hscore = dict()

@wu.timed()
def asudock(sym, coords, scalings=[1.0], ignored_aas='CP', use_rosetta=False, **kw):

   kw = wu.Bunch(kw)
   assert not np.any(np.isnan(coords))
   assert not np.any(coords == 9e9)
   origbody = rp.Body(coords, extract_chain=0, ignored_aas=ignored_aas, use_rosetta=use_rosetta, **kw)
   wu.checkpoint(kw, 'bodies')
   cellsize = get_cellsize(sym, origbody)
   wu.checkpoint(kw, 'read cellsize')

   results = list()
   for iscale, scale in enumerate(scalings):
      csize = cellsize * scale
      xdelta = wu.htrans((scale - 1) * origbody.com())
      ic(iscale, scale, xdelta[:3, 3])
      body = origbody.copy_xformed(xdelta, **kw)
      bodies0, bodies1 = get_body_pairs(sym, body, csize, **kw)
      primaryframes = wu.sym.frames(sym, cells=None, cellsize=csize)
      clashframes = wu.sym.frames(sym, cells=3, cellsize=csize, center=body.com(), asucen=body.com(), xtalrad=0.5, **kw)
      x2asymcen = wu.htrans(-body.com())
      result = rp.search.make_asym(
         [bodies0, bodies1],
         hscore=get_hscore(**kw),
         sampler=get_sampler(sym, **kw),
         frames=primaryframes,
         clashframes=clashframes,
         x2asymcen=x2asymcen,
         sym=sym,
         **kw,
      )
      # asux = result.xforms.data
      asux = wu.hxform(xdelta, result.xforms.data)
      result.data['asuxforms'] = (['model', 'hrow', 'hcol'], asux)
      if not len(result):
         continue
      maxsc = np.max(result.scores.data)
      ic(scale, maxsc)
      result.data.attrs['sym'] = sym
      celldat = np.tile([csize, csize, csize, 90, 90, 90], len(result.data.scores)).reshape(-1, 6)
      result.data['unitcell'] = (['model', 'celldim'], celldat)
      results.append(result)
   result = rp.concat_results(results)
   if result is None: return None
   print(result, flush=True)
   kw.output_prefix = f'{kw.output_prefix}_rp'
   kw.output_suffix = f'_scale{scale:05.3f}'
   # body.dump_pdb(f'{pre}_init.pdb', symframes=frames)
   # header = wu.sym.xtal(sym).cryst1(cellsize=cellsize)
   # for i, b in enumerate(result.bodies):
   # ic(b, id(b))
   # b[0].dump_pdb(f'body{i}.pdb')
   if result.scores[0] > 10:
      # result.dump_pdbs_top_score(**kw.sub(output_suffix='_sym'))
      result.dump_pdbs_top_score(symframes='primary', **kw.sub(output_suffix='_asym'))
   wu.checkpoint(kw, 'dump_pdbs')

   return result

def asudock_main(**kw):
   kw = get_asudock_config(kw)
   sym = kw.architecture
   # sampler = get_sampler(sym, **kw)
   wu.checkpoint(kw, 'init')
   results = list()

   # pdb = wu.readpdb(kw.inputs1[0])
   # crd = pdb.ncaco(splitchains=True)
   # for fname in kw.inputs1 * 10:

   for fname in kw.inputs1:
      pdb = wu.readpdb(fname)
      pdb.dump_pdb('ref.pdb')
      crd = pdb.ncaco(splitchains=True)
      assert not np.any(crd == 9e9)
      if not kw.skip_errors:
         result = asudock(sym, crd, **kw)
      else:
         try:
            result = asudock(sym, crd, **kw)
         except (ValueError, AssertionError) as e:
            ic(e)
            ic('ERROR fail on', fname)
            continue
      results.append(result)

   # results = rp.concat_results(results)
   # results.dump_pdbs_top_score(10, symframes=frames, output_prefix='top10', dump_input=True, header=header)
   kw.timer.report()

@wu.timed
def get_cellsize(sym, body):
   cellsize = None
   if body.crystinfo is not None and body.crystinfo.spacegroup.strip() != 'P 1':
      ci = body.crystinfo
      assert sym.split('_')[0].replace(' ', '') == ci.spacegroup.replace(' ', '')
      assert max(ci.cell) == min(ci.cell)
      cellsize = ci.cell[0]
   if cellsize is None:
      x = wu.sym.xtal(sym)
      if body.rawpdb is None:
         coords = body.rawcoords
      else:
         coords = body.rawpdb.ca()
         coords = coords.reshape(body.rawpdb.nchain, -1, 3)
      _crd, cellsize = x.fit_coords(coords)
      # cellsize, _ = x.fit_coords(coords, noshift=True, mcsteps=100, cellsize=50)
   return cellsize

def get_body_pairs(sym, body, cellsize, preserve_contacts=False, **kw):
   origprimaryframes = wu.sym.frames(sym, cells=None, cellsize=cellsize)
   bodies0 = [body.copy() for b in origprimaryframes[:-1]]
   bodies1 = [body.copy() for b in origprimaryframes[1:]]
   if preserve_contacts:
      rsets0, rsets1 = set_allowed_residues_by_current_neighbors(body=body, frames=origprimaryframes, **kw)
      for i in range(len(origprimaryframes) - 1):
         bodies0[i].set_allowed_residues(required_res_sets=[rsets0[i]])
         bodies1[i].set_allowed_residues(required_res_sets=[rsets1[i]])
   return bodies0, bodies1

@wu.timed()
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
         if len(required_res_set0) >= 10 and len(required_res_set1) >= 10:
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

def get_hscore(hscore_files, max_pair_dist=8, **kw):
   global _hscore
   key = tuple(hscore_files), max_pair_dist
   if key not in _hscore:
      _hscore[key] = rp.RpxHier(hscore_files, max_pair_dist, **kw)
   return _hscore[key]

def main():
   kw = rp.options.get_cli_args()
   # kw.scalings = [0.95, 1.0, 1.05]
   # kw.scalings = [0.97, 1, 1.06]
   # kw.scalings = [0.95]
   kw.scalings = [1.05]
   kw.nout_top = 10
   kw.cartbound = 10
   kw.limit_rotation = 0.
   kw.limit_translation = 4
   kw.oriresl = 20
   kw.cartcells = 5 if kw.limit_rotation > 0 else 10
   kw.beam_size = 10e3
   kw.ignored_aas = 'CP'
   asudock_main(**kw)
   print('asudock_main.py DONE')

# def asudock_main

def get_sampler(sym, cartbound=15, cartcells=5, oriresl=30, limit_rotation=0.2, **kw):
   cartlb = np.array([-cartbound, -cartbound, -cartbound])
   cartub = np.array([+cartbound, +cartbound, +cartbound])
   cartbs = np.array([cartcells, cartcells, cartcells], dtype='i')
   if wu.sym.ndim(sym) == 3:
      if limit_rotation > 0:
         sampler = rp.sampling.XformHier_f4(cartlb, cartub, cartbs, oriresl)
      else:
         sampler = rp.sampling.CartHier3D_f4(cartlb, cartub, cartbs)
   elif wu.sym.ndim(sym) == 2:
      sampler = rp.sampling.OriCart2Hier_f4(cartlb[:2], cartub[:2], cartbs[:2], oriresl)
   elif wu.sym.ndim(sym) == 0:
      raise NotImplementedError('need to audit cages')
   return sampler

def get_asudock_config(kw):
   kwdefault = rp.options.defaults()
   kwdefault.preserve_contacts = True
   kwdefault.ignored_aas = 'CP'
   kw = kwdefault.sub(kw)
   kw.timer = wu.Timer()
   kw.filter_sscount = dict(sscount_near=dict(
      type='filter_sscount',
      confidence=True,
      min_helix_length=4,
      min_sheet_length=3,
      min_loop_length=1,
      max_dist=8,
      min_element_resis=3,
      sstype="EH",
      min_ss_count=4,
      strict=False,
   ))
   kw = kw.sub(filter_sscount=None)

   kw.wts = wu.Bunch(ncontact=0.01, rpx=1.0)
   kw.max_bb_redundancy = 1.0
   # kw.ignored_aas = 'CP'
   kw.hscore_files = 'ilv_h'
   # kw.hscore_files = 'ilv_h/1000'
   return kw

if __name__ == '__main__':
   main()
