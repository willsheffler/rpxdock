#! /home/sheffler/.conda/envs/rpxdock/bin/python

import logging, itertools, concurrent, tqdm, rpxdock as rp
import numpy as np

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

## All dock_cyclic, dock_onecomp, and dock_multicomp do similar things
def dock_cyclic(hscore, inputs, architecture, **kw):
   kw = rp.Bunch(kw)
   bodies = [
      rp.Body(inp, allowed_res=allowedres, **kw)
      for inp, allowedres in zip(kw.inputs1, kw.allowed_residues1)
   ]
   exe = concurrent.futures.ProcessPoolExecutor
   # exe = rp.util.InProcessExecutor
   with exe(kw.ncpu) as pool:
      futures = list()
      # where the magic happens
      for ijob, bod in enumerate(bodies):
         futures.append(
            pool.submit(
               rp.search.make_cyclic,
               bod,
               architecture.upper(),
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
   kw = rp.Bunch(kw)
   spec = get_spec(kw.architecture)
   crtbnd = kw.cart_bounds[0]

   # double normal resolution, cuz why not?
   if kw.docking_method == 'grid':
      flip=list(spec.flip_axis[:3])
      if not kw.flip_components[0]:
         flip = None
      sampler = rp.sampling.grid_sym_axis(
         cart=np.arange(crtbnd[0], crtbnd[1], kw.grid_resolution_cart_angstroms),
         ang=np.arange(0, 360 / spec.nfold, kw.grid_resolution_ori_degrees),
         axis=spec.axis,
         flip=flip
         )
      search = rp.grid_search
   else:
      if spec.type == 'mirrorlayer':
         sampler = rp.sampling.hier_mirror_lattice_sampler(spec, resl=10, angresl=10, **kw)
      else:
         sampler = rp.sampling.hier_axis_sampler(spec.nfold, lb=crtbnd[0], ub=crtbnd[1], resl=5,
                                                 angresl=5, axis=spec.axis, flipax=spec.flip_axis, **kw)
      search = rp.hier_search

   # pose info and axes that intersect
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
   kw = rp.Bunch(kw)
   spec = get_spec(kw.architecture)
   sampler = rp.sampling.hier_multi_axis_sampler(spec, **kw)
   logging.info(f'num base samples {sampler.size(0):,}')

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
   kw = rp.Bunch(kw)
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
      sampler = rp.sampling.hier_axis_sampler(kw.nfold, lb=crtbnd[0], ub=crtbnd[1], resl=10,
                                              angresl=10)
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

def dock_axel(hscore, **kw):
   arg = rp.Bunch(kw)
   spec = get_spec(arg.architecture)
   
   if arg.docking_method.lower() == 'hier':
       sampler1 = rp.sampling.hier_axis_sampler(spec.nfold[0],  lb=0, ub=100, resl=5, angresl=5)
       sampler2 = rp.sampling.ZeroDHier([np.eye(4),rp.homog.hrot([1,0,0],180)])
       sampler = rp.sampling.CompoundHier(sampler2, sampler1)
       logging.info(f'num base samples {sampler.size(0):,}')
       search = rp.hier_search
   elif arg.docking_method.lower() =='grid':
       sampler1 = rp.sampling.grid_sym_axis(
         cart=np.arange(0, 100, arg.grid_resolution_cart_angstroms),
         ang=np.arange(0, 360 / spec.nfold[0], arg.grid_resolution_ori_degrees),
         axis=spec.axis[0],
         )
       assert sampler1.ndim == 3
       shape = (2,)+sampler1.shape
       sampler = np.zeros(shape=shape)
       sampler[0] = sampler1
       sampler[1] = np.eye(4)
       sampler = np.swapaxes(sampler,0,1)
       logging.info(f'num base samples {sampler.shape}')
       search = rp.grid_search
   else:
       raise ValueError(f'not compatible with docking method {arg.dock_method}')   
   
   bodies = [[rp.Body(fn, **arg) for fn in inp] for inp in arg.inputs]
   assert len(bodies) == spec.num_components

   exe = concurrent.futures.ProcessPoolExecutor
   with exe(arg.ncpu) as pool:
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
               **arg,
            ))
         futures[-1].ijob = ijob
      result = [None] * len(futures)
      for f in tqdm.tqdm(concurrent.futures.as_completed(futures), total=len(futures)):
         result[f.ijob] = f.result()
   result = rp.concat_results(result)
   return result

def dock_plug(hscore, **kw):
   arg = rp.Bunch(kw)
   arg.plug_fixed_olig = True

   arch = arg.architecture
   #arg.sym = arch.split('_')[1]
   arg.nfold = int(arch.split('_')[1][-1])

   cb = arg.cart_bounds[0]
   if not cb: cb = [-100, 100]
   if arg.docking_method.lower() == 'grid':
      search = rp.grid_search
      crt_smap = np.arange(cb[0], cb[1] + 0.001, arg.grid_resolution_cart_angstroms)
      ori_samp = np.arange(-180 / arg.nfold, 180 / arg.nfold - 0.001,
                           arg.grid_resolution_ori_degrees)
      sampler = rp.sampling.grid_sym_axis(crt_smap, ori_samp, axis=[0, 0, 1], flip=[0, 1, 0])
      logging.info(f'docking samples per splice {len(sampler)}')
   elif arg.docking_method.lower() == 'hier':
      search = rp.hier_search
      sampler = rp.sampling.hier_axis_sampler(arg.nfold, lb=cb[0], ub=cb[1])
      logging.info(f'docking possible samples per splice {sampler.size(4)}')
   else:
      raise ValueError(f'unknown search dock_method {arg.dock_method}')

   logging.info(f'num base samples {sampler.size(0):,}')

   plug_bodies = [rp.Body(inp, which_ss="H", **arg) for inp in arg.inputs1]
   hole_bodies = [rp.Body(inp, sym=3, which_ss="H", **arg) for inp in arg.inputs2]

   #assert len(bodies) == spec.num_components

   exe = concurrent.futures.ProcessPoolExecutor
   # exe = rp.util.InProcessExecutor
   with exe(arg.ncpu) as pool:
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
               **arg,
            ))
         futures[-1].ijob = ijob
      result = [None] * len(futures)
      for f in tqdm.tqdm(concurrent.futures.as_completed(futures), total=len(futures)):
         result[f.ijob] = f.result()
   result = rp.concat_results(result)
   return result

def main():
   kw = get_rpxdock_args()
   rp.options.print_options(kw)
   print(f'{" RUNNING dock.py:main ":=^80}')

   logging.info(f'weights: {kw.wts}')

   hscore = rp.CachedProxy(rp.RpxHier(kw.hscore_files, **kw))
   arch = kw.architecture


   # TODO: redefine archs WHS or others with a monster list of if statements
   if arch.startswith('C'):
      result = dock_cyclic(hscore, **arg)
   elif len(arch) == 2 or (arch[0] == 'D' and arch[2] == '_'):
      result = dock_onecomp(hscore, **arg)
   elif arch.startswith('PLUG'):
      result = dock_plug(hscore, **arg)
   elif arch.startswith('AXEL_'):
      result = dock_axel(hscore, **arg)
   else:
      result = dock_multicomp(hscore, **kw)

   print(result)
   if kw.dump_pdbs:
      result.dump_pdbs_top_score(hscore=hscore, **kw)
      result.dump_pdbs_top_score_each(hscore=hscore, **kw)
   if not kw.suppress_dump_results:
      rp.util.dump(result, kw.output_prefix + '_Result.pickle')

if __name__ == '__main__':
   main()
