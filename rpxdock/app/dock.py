#! /home/sheffler/.conda/envs/rpxdock/bin/python

import logging, itertools, concurrent, tqdm, os, sys
import rpxdock as rp
import numpy as np
from willutil import Bunch

def get_rpxdock_args():
   kw = rp.options.get_cli_args()
   if not kw.architecture: raise ValueError("architecture must be specified")
   return kw

def get_spec(arch):
   arch = arch.upper()
   if arch.startswith('P') and not arch.startswith('PLUG'):
      sym = arch.split('_')[0]
      component_nfold = arch.split('_')[1]
      ismirror = sym[-1] == 'M'
      if len(component_nfold) == 1 and ismirror:
         spec = rp.search.DockSpec1CompMirrorLayer(arch)
      elif len(component_nfold) == 1:
         spec = rp.search.DockSpec1CompLayer(arch)
      elif len(component_nfold) == 2:
         spec = rp.search.DockSpec2CompLayer(arch)
      elif len(component_nfold) == 3:
         spec = rp.search.DockSpec3CompLayer(arch)
      else:
         raise ValueError('number of conponents must be 1, 2 or 3')
   elif len(arch) == 2 or (arch[0] == 'D' and arch[2] == '_'):
      spec = rp.search.DockSpec1CompCage(arch)
   elif arch.startswith('AXEL_'):
      spec = rp.search.DockSpecAxel(arch)
   else:
      spec = rp.search.DockSpec2CompCage(arch)
   return spec

def gcd(a, b):
   while b > 0:
      a, b = b, a % b
   return a

## All dock_cyclic, dock_onecomp, and dock_multicomp do similar things

def dock_asym(hscore, **kw):
   kw = Bunch(kw, _strict=False)
   if kw.cart_bounds[0]:
      crtbnd = kw.cart_bounds[0]
      extent = crtbnd[1]
      cartlb = np.array([crtbnd[0], crtbnd[0], crtbnd[0]])
      cartub = np.array([crtbnd[1], crtbnd[1], crtbnd[1]])
   else:
      extent = 100
      cartlb = np.array([-extent] * 3)
      cartub = np.array([extent] * 3)

   cartbs = np.array([kw.cart_resl] * 3, dtype="i")

   logging.debug('dock_asym bounds')
   cartub = [30, 30, 30]
   kw.ori_resl = 60
   logging.debug(f"  cartlb {cartlb}")
   logging.debug(f"  cartub {cartub}")
   logging.debug(f"  cartbs {cartbs}")
   logging.debug(f"  ori_resl {kw.ori_resl}")

   sampler = rp.sampling.XformHier_f4(cartlb, cartub, cartbs, kw.ori_resl)
   logging.info(f'num base samples {sampler.size(0):,}')

   bodies = [[rp.Body(fn, allowed_res=ar2, **kw)
              for fn, ar2 in zip(inp, ar)]
             for inp, ar in zip(kw.inputs, kw.allowed_residues)]

   exe = concurrent.futures.ProcessPoolExecutor
   # exe = rp.util.InProcessExecutor
   with exe(kw.ncpu) as pool:
      # if 0:
      futures = list()
      # TODO: why is this a product? not same behavior as other protocols
      # should ask ppl if this is desired behavior
      for ijob, bod in enumerate(itertools.product(*bodies)):
         futures.append(
            pool.submit(
               rp.search.make_asym,
               bod,
               hscore,
               sampler,
               rp.hier_search,
               **kw,
            ))
         futures[-1].ijob = ijob
      results = [None] * len(futures)
      for f in tqdm.tqdm(concurrent.futures.as_completed(futures), total=len(futures)):
         results[f.ijob] = f.result()

      # rp.dump(results, 'tmp.pickle')
      # results = rp.load('tmp.pickle')
      result = rp.concat_results(results)

   # print(bodies)
   # result = rp.search.make_asym(
   #    [bodies[0][0], bodies[1][0]],
   #    hscore,
   #    sampler,
   #    rp.hier_search,
   #    **kw,
   # )
   # result = rp.concat_results([result])

   return result

def dock_cyclic(hscore, **kw):
   kw = Bunch(kw, _strict=False)
   bodies = [
      rp.Body(inp, allowed_res=allowedres, **kw)
      for inp, allowedres in zip(kw.inputs1, kw.allowed_residues1)
   ]
   # exe = concurrent.futures.ProcessPoolExecutor
   exe = rp.util.InProcessExecutor
   with exe(kw.ncpu) as pool:
      futures = list()
      # where the magic happens
      for ijob, bod in enumerate(bodies):
         futures.append(
            pool.submit(
               rp.search.make_cyclic,
               bod,
               kw.architecture.upper(),
               hscore,
               **kw,
            ))
         futures[-1].ijob = ijob
      result = [None] * len(futures)
      for f in tqdm.tqdm(concurrent.futures.as_completed(futures), total=len(futures)):
         result[f.ijob] = f.result()
   result = rp.concat_results(result)

   # result = rp.search.make_cyclic(body, architecture.upper(), hscore, **kw)

   return result

def dock_onecomp(hscore, **kw):
   kw = Bunch(kw, _strict=False)
   spec = get_spec(kw.architecture)
   crtbnd = kw.cart_bounds[0]

   # If necessary determine accessibility and flip restriction first
   make_poselist = False
   for inp_pair in kw.term_access:
      if not make_poselist: make_poselist = any(True in pair for pair in inp_pair)
   if make_poselist or not all(None in pair for pair in kw.termini_dir):
      poses, og_lens = rp.rosetta.helix_trix.init_termini(make_poselist, **kw)

   # double normal resolution, cuz why not?
   if kw.docking_method == 'grid':
      flip = list(spec.flip_axis[:3])
      force_flip = False if kw.force_flip is None else kw.force_flip[0]
      if not kw.flip_components[0]:
         flip = None

      sampler = rp.sampling.grid_sym_axis(
         cart=np.arange(crtbnd[0], crtbnd[1], kw.grid_resolution_cart_angstroms), ang=np.arange(
            0, 360 / spec.nfold, kw.grid_resolution_ori_degrees), axis=spec.axis, flip=flip,
         force_flip=force_flip)

      search = rp.grid_search
   else:
      if spec.type == 'mirrorlayer':
         sampler = rp.sampling.hier_mirror_lattice_sampler(spec, resl=10, angresl=10, **kw)
      else:

         sampler = rp.sampling.hier_axis_sampler(
            spec.nfold,
            lb=crtbnd[0],
            ub=crtbnd[1],
            resl=5,
            angresl=5,
            axis=spec.axis,
            flipax=spec.flip_axis,
            **kw,
         )

      search = rp.hier_search

   # pose info and axes that intersect.
   # Use list of modified poses to make bodies if such list exists
   if make_poselist:
      assert len(poses) == len(og_lens)
      bodies = [
         rp.Body(pose1, allowed_res=allowedres, modified_term=modterm, og_seqlen=og_seqlen,
                 **kw) for pose1, allowedres, modterm, og_seqlen in zip(
                    poses[0], kw.allowed_residues1, kw.term_access1, og_lens[0])
      ]
   else:
      bodies = [
         rp.Body(inp, allowed_res=allowedres, **kw)
         for inp, allowedres in zip(kw.inputs1, kw.allowed_residues1)
      ]

   exe = concurrent.futures.ProcessPoolExecutor
   # exe = rp.util.InProcessExecutor
   with exe(kw.ncpu) as pool:
      futures = list()
      for ijob, bod in enumerate(bodies):
         futures.append(
            pool.submit(
               rp.search.make_onecomp,
               bod,
               spec,
               hscore,
               search,
               sampler,
               **kw,
            ))
         futures[-1].ijob = ijob
      result = [None] * len(futures)
      for f in tqdm.tqdm(concurrent.futures.as_completed(futures), total=len(futures)):
         result[f.ijob] = f.result()

   result = rp.concat_results(result)
   return result
   # result = rp.search.make_onecomp(bodyC3, spec, hscore, rp.hier_search, sampler, **kw)

def dock_multicomp(hscore, **kw):
   kw = Bunch(kw, _strict=False)
   spec = get_spec(kw.architecture)

   # <<<<<<< HEAD
   # sampler = rp.sampling.hier_multi_axis_sampler(spec, **kw)
   # sampler1 = rp.sampling.RotCart1Hier_f4(
   #    cart_bounds[i, 0],
   #    cart_bounds[i, 1],
   #    cart_nstep[i],
   #    0,
   #    ang[i],
   #    ang_nstep[i],
   #    spec.axis[i][:3],
   # )
   # sampler2 = rp.ZeroDHier([np.eye(4)])
   # sampler = rp.sampling.CompoundHier(sampler1, sampler2)
   # =======

   # Determine accessibility and flip restriction before we make sampler
   make_poselist = False
   for inp_pair in kw.term_access:
      if not make_poselist: make_poselist = any(True in pair for pair in inp_pair)
   if make_poselist or not all(None in pair for pair in kw.termini_dir):
      poses, og_lens = rp.rosetta.helix_trix.init_termini(make_poselist, **kw)

   sampler = rp.sampling.hier_multi_axis_sampler(spec, **kw)
   # >>>>>>> master
   logging.info(f'num base samples {sampler.size(0):,}')

   # Use list of modified poses to make bodies if such list exists
   if make_poselist:
      assert len(poses) == len(og_lens)
      bodies = [[
         rp.Body(pose2, og_seqlen=og2, modified_term=modterm2, **kw)
         for pose2, og2, modterm2 in zip(pose1, og1, modterm1)
      ]
                for pose1, og1, modterm1 in zip(poses, og_lens, kw.term_access)]

   else:
      bodies = [[rp.Body(fn, allowed_res=ar2, **kw)
                 for fn, ar2 in zip(inp, ar)]
                for inp, ar in zip(kw.inputs, kw.allowed_residues)]
   assert len(bodies) == spec.num_components

   exe = concurrent.futures.ProcessPoolExecutor
   # exe = rp.util.InProcessExecutor
   with exe(kw.ncpu) as pool:
      futures = list()
      for ijob, bod in enumerate(itertools.product(*bodies)):
         futures.append(
            pool.submit(
               rp.search.make_multicomp,
               bod,
               spec,
               hscore,
               rp.hier_search,
               sampler,
               **kw,
            ))
         futures[-1].ijob = ijob
      result = [None] * len(futures)
      for f in tqdm.tqdm(concurrent.futures.as_completed(futures), total=len(futures)):
         result[f.ijob] = f.result()
   result = rp.concat_results(result)
   return result

def dock_plug(hscore, **kw):
   kw = Bunch(kw, _strict=False)
   kw.plug_fixed_olig = True

   arch = kw.architecture
   kw.nfold = int(arch.split('_')[1][-1])

   crtbnd = kw.cart_bounds[0]
   if not crtbnd: crtbnd = [-100, 100]
   if kw.docking_method.lower() == 'grid':
      search = rp.grid_search
      crt_smap = np.arange(crtbnd[0], crtbnd[1] + 0.001, kw.grid_resolution_cart_angstroms)
      ori_samp = np.arange(-180 / kw.nfold, 180 / kw.nfold - 0.001,
                           kw.grid_resolution_ori_degrees)
      sampler = rp.sampling.grid_sym_axis(crt_smap, ori_samp, axis=[0, 0, 1], flip=[0, 1, 0])
      logging.info(f'docking samples per splice {len(sampler)}')
   elif kw.docking_method.lower() == 'hier':
      search = rp.hier_search
      sampler = rp.sampling.hier_axis_sampler(lb=crtbnd[0], ub=crtbnd[1], resl=10, angresl=10,
                                              **kw)
      logging.info(f'docking possible samples per splice {sampler.size(4)}')
   else:
      raise ValueError(f'unknown search dock_method {kw.dock_method}')

   plug_bodies = [
      rp.Body(inp, which_ss="H", allowed_res=allowedres, **kw)
      for inp, allowedres in zip(kw.inputs1, kw.allowed_residues1)
   ]
   hole_bodies = [
      rp.Body(inp, which_ss="H", allowed_res=allowedres, **kw)
      for inp, allowedres in zip(kw.inputs2, kw.allowed_residues2)
   ]

   #assert len(bodies) == spec.num_components

   exe = concurrent.futures.ProcessPoolExecutor
   # exe = rp.util.InProcessExecutor
   with exe(kw.ncpu) as pool:
      futures = list()
      for ijob, bod in enumerate(itertools.product(hole_bodies)):
         hole = hole_bodies[ijob]
         plug = plug_bodies[ijob]
         futures.append(
            pool.submit(
               rp.search.make_plugs,
               plug,
               hole,
               hscore,
               search,
               sampler,
               **kw,
            ))
         futures[-1].ijob = ijob
      result = [None] * len(futures)
      for f in tqdm.tqdm(concurrent.futures.as_completed(futures), total=len(futures)):
         result[f.ijob] = f.result()
   result = rp.concat_results(result)
   return result

# Docks two comopnents against one another, algined along axis of sym
# Component one (inputs1) remains stationary during docking, but can flip
# Component 2 (inputs2) rotates about axis, and can also flip
def dock_axel(hscore, **kw):
   kw = Bunch(kw, _strict=False)
   spec = get_spec(kw.architecture)
   flip = list(spec.flip_axis)

   if kw.docking_method.lower() == 'hier':
      if not kw.flip_components[0]:
         flip[0] = None
      if spec.nfold[0] == spec.nfold[1]:
         sampler1 = rp.sampling.hier_axis_sampler(
            spec.nfold[0],
            lb=0,
            ub=100,
            resl=5,
            angresl=5,
            axis=[0, 0, 1],
            flipax=flip[0],
            flip_components=kw.flip_components[0],
         )
      else:
         sampler1 = rp.sampling.hier_axis_sampler(
            spec.nfold[0] * spec.nfold[1] / gcd(spec.nfold[0], spec.nfold[1]),
            lb=0,
            ub=100,
            resl=5,
            angresl=5,
            axis=[0, 0, 1],
            flipax=flip[0],
            flip_components=kw.flip_components[0],
         )
      sampler2 = rp.sampling.ZeroDHier([np.eye(4), rp.homog.hrot([1, 0, 0], 180)])
      if not kw.flip_components[-1]:  #Flipping for second input
         sampler2 = rp.sampling.ZeroDHier(np.eye(4))
      sampler = rp.sampling.CompoundHier(sampler2, sampler1)
      logging.info(f'num base samples {sampler.size(0):,}')
      search = rp.hier_search

   elif kw.docking_method.lower() == 'grid':
      flip[0] = list(spec.flip_axis[0, :3])
      if not kw.flip_components[0]:
         flip[0] = None
      if spec.nfold[0] == spec.nfold[1]:
         sampler1 = rp.sampling.grid_sym_axis(
            cart=np.arange(0, 100, kw.grid_resolution_cart_angstroms),
            ang=np.arange(0, 360 / spec.nfold[0], kw.grid_resolution_ori_degrees),
            axis=spec.axis[0],
            flip=flip[0],
         )
      else:
         sampler1 = rp.sampling.grid_sym_axis(
            cart=np.arange(0, 100, kw.grid_resolution_cart_angstroms), ang=np.arange(
               0, 360 / (spec.nfold[0] * spec.nfold[1] / gcd(spec.nfold[0], spec.nfold[1])),
               kw.grid_resolution_ori_degrees), axis=spec.axis[0], flip=flip[0])
      assert sampler1.ndim == 3
      if not kw.flip_components[-1]:  # For second input
         shape = (2, *sampler1.shape)
         sampler = np.zeros(shape=shape)
         sampler[0, ] = np.eye(4)
         sampler[1, ] = sampler1
      else:
         shape = (2, 2 * len(sampler1), 4, 4)
         sampler = np.zeros(shape=shape)
         sampler[0, :len(sampler1)] = np.eye(4)
         sampler[0, len(sampler1):] = rp.homog.hrot([1, 0, 0], 180)
         sampler[1, :len(sampler1)] = sampler1
         sampler[1, len(sampler1):] = sampler1
      sampler = np.swapaxes(sampler, 0, 1)
      logging.info(f'num base samples {sampler.shape}')
      search = rp.grid_search

   else:
      raise ValueError(f'not compatible with docking method {kw.dock_method}')

   bodies = [[rp.Body(fn, allowed_res=ar2, **kw)
              for fn, ar2 in zip(inp, ar)]
             for inp, ar in zip(kw.inputs, kw.allowed_residues)]
   assert len(bodies) == spec.num_components
   #   bodies = [[rp.Body(fn, **kw) for fn in inp] for inp in kw.inputs]
   #   assert len(bodies) == spec.num_components

   exe = concurrent.futures.ProcessPoolExecutor
   with exe(kw.ncpu) as pool:
      futures = list()
      for ijob, bod in enumerate(itertools.product(*bodies)):
         futures.append(
            pool.submit(
               rp.search.make_multicomp,
               bod,
               spec,
               hscore,
               search,
               sampler,
               **kw,
            ))
         futures[-1].ijob = ijob
      result = [None] * len(futures)
      for f in tqdm.tqdm(concurrent.futures.as_completed(futures), total=len(futures)):
         result[f.ijob] = f.result()
   result = rp.concat_results(result)
   return result

def check_result_files_exist(kw):
   kw = Bunch(kw)
   tarfname = kw.output_prefix + '_Result.txz'
   picklefname = kw.output_prefix + '_Result.pickle'
   if not kw.suppress_dump_results:
      if kw.save_results_as_tarball:
         if os.path.exists(tarfname) and not kw.overwrite_existing_results:
            logging.info(f"Results File Exists: {str(tarfname)}")
            logging.info('Move files or use --overwrite_existing_results')
            sys.exit()
      if kw.save_results_as_pickle:
         if os.path.exists(picklefname) and not kw.overwrite_existing_results:
            logging.info(f"Results File Exists: {str(picklefname)}")
            logging.info('Move files or use --overwrite_existing_results')
            sys.exit()

def main():
   kw = get_rpxdock_args()
   logging.info(f'{" RUNNING dock.py:main ":=^80}')
   #logging.info(f'weights: {kw.wts}')
   rp.options.print_options(kw)

   hscore = rp.CachedProxy(rp.RpxHier(kw.hscore_files, **kw))
   arch = kw.architecture

   check_result_files_exist(kw)

   # TODO commit to master AK
   # sym, comp = arch.split('_')
   # TODO: redefine archs WHS or others with a monster list of if statements
   if arch.startswith('C'):
      result = dock_cyclic(hscore, **kw)
   elif arch == 'ASYM':
      result = dock_asym(hscore, **kw)
   elif len(arch) == 2 or (arch[0] == 'D' and arch[2] == '_'):
      result = dock_onecomp(hscore, **kw)
   elif arch.startswith('PLUG'):
      result = dock_plug(hscore, **kw)
   elif arch.startswith('AXEL_'):
      result = dock_axel(hscore, **kw)
   else:
      result = dock_multicomp(hscore, **kw)

   if kw.dump_pdbs:
      result.dump_pdbs_top_score(hscore=hscore, **kw)
      result.dump_pdbs_top_score_each(hscore=hscore, **kw)
   if not kw.suppress_dump_results:

      if kw.save_results_as_pickle:
         picklefname = kw.output_prefix + '_Result.pickle'
         rp.util.dump(result, picklefname)
         logging.info(f"saved result to {str(picklefname)}")

      if kw.save_results_as_tarball:
         tarfname = kw.output_prefix + '_Result.txz'
         rp.search.result_to_tarball(result, tarfname, overwrite=True)
         logging.info(f"saved result to {str(tarfname)}")

         #print('try to read')
         #result2 = rp.search.result_from_tarball(tarfname)
         #assert result == result2
         #assert result2.sym
         ## print(result2)
         #result2.dump_pdbs()

      if True:
         summaryfname = kw.output_prefix + '_Summary.txt'
         logging.info(f"saving summary to {str(summaryfname)}")
         rp.search.result_to_summary()
         #rp.search.result_to_summary(result, summaryfname, overwrite=True)
         logging.info(f"saved summary to {str(summaryfname)}")

if __name__ == '__main__':
   main()
   logging.info('DONE')
