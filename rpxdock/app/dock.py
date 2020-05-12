#! /home/sheffler/.conda/envs/rpxdock/bin/python

import logging, itertools, concurrent, tqdm, rpxdock as rp

def get_rpxdock_args():
   arg = rp.options.get_cli_args()
   if not arg.architecture: raise ValueError("architecture must be specified")
   return arg

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
   else:
      spec = rp.search.DockSpec2CompCage(arch)
   return spec

## All dock_cyclic, dock_onecomp, and dock_multicomp do similar things
def dock_cyclic(hscore, inputs, architecture, **kw):
   ## bunch is a dictionary of things
   arg = rp.Bunch(kw)
   ## bodies hold all pose info and axis intersect info
   bodies = [rp.Body(inp, **arg) for inp in arg.inputs1]

   ## start the parallel processes
   exe = concurrent.futures.ProcessPoolExecutor
   # exe = rp.util.InProcessExecutor
   with exe(arg.ncpu) as pool:
      futures = list()
      # where the magic happens
      for ijob, bod in enumerate(bodies):
         futures.append(
            pool.submit(
               rp.search.make_cyclic,
               bod,
               architecture.upper(),
               hscore,
               **arg,
            ))
         futures[-1].ijob = ijob
      result = [None] * len(futures)
      for f in tqdm.tqdm(concurrent.futures.as_completed(futures), total=len(futures)):
         result[f.ijob] = f.result()
   result = rp.concat_results(result)

   # result = rp.search.make_cyclic(body, architecture.upper(), hscore, **arg)

   return result

def dock_onecomp(hscore, **kw):
   arg = rp.Bunch(kw)
   # for 1comp, len(arch) == 2
   spec = get_spec(arg.architecture)
   # double normal resolution, cuz why not?
   # mirrorlayer for 1comp xtals and stuff with P architectures
   if spec.type == 'mirrorlayer':
      sampler = rp.sampling.hier_mirror_lattice_sampler(spec, resl=10, angresl=10, **arg)
   else:
      #TODO: Change sampler to grid_sym_axis and switch rp.hier_search to rp.grid_search (if arg.docking_method == "grid" or "hier") AK/EY
      sampler = rp.sampling.hier_axis_sampler(spec.nfold, lb=0, ub=100, resl=5, angresl=5,
                                              axis=spec.axis, flipax=spec.flip_axis)
   # pose info and axes that intersect
   bodies = [rp.Body(inp, **arg) for inp in arg.inputs1]

   exe = concurrent.futures.ProcessPoolExecutor
   # exe = rp.util.InProcessExecutor
   with exe(arg.ncpu) as pool:
      futures = list()
      for ijob, bod in enumerate(bodies):
         futures.append(
            pool.submit(
               rp.search.make_onecomp,
               bod,
               spec,
               hscore,
               rp.hier_search,
               sampler,
               **arg,
            ))
         futures[-1].ijob = ijob
      result = [None] * len(futures)
      for f in tqdm.tqdm(concurrent.futures.as_completed(futures), total=len(futures)):
         result[f.ijob] = f.result()
   result = rp.concat_results(result)
   return result
   # result = rp.search.make_onecomp(bodyC3, spec, hscore, rp.hier_search, sampler, **arg)

def dock_multicomp(hscore, **kw):
   arg = rp.Bunch(kw)
   spec = get_spec(arg.architecture)
   sampler = rp.sampling.hier_multi_axis_sampler(spec, **arg)
   logging.info(f'num base samples {sampler.size(0):,}')

   bodies = [[rp.Body(fn, **arg) for fn in inp] for inp in arg.inputs]
   assert len(bodies) == spec.num_components

   exe = concurrent.futures.ProcessPoolExecutor
   # exe = rp.util.InProcessExecutor
   with exe(arg.ncpu) as pool:
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
   # What gets all the shit done
   arg = get_rpxdock_args()
   logging.info(f'weights: {arg.wts}')

   hscore = rp.CachedProxy(rp.RpxHier(arg.hscore_files, **arg))
   arch = arg.architecture

   # TODO commit to master AK
   #sym, comp = arch.split('_')

   # TODO: redefine archs WHS or others with a monster list of if statements
   if arch.startswith('C'):
      result = dock_cyclic(hscore, **arg)
   elif len(arch) == 2 or (arch[0] == 'D' and arch[2] == '_'):
      result = dock_onecomp(hscore, **arg)
   elif arch.startswith('PLUG'):
      result = dock_plug(hscore, **arg)
   else:
      result = dock_multicomp(hscore, **arg)

   print(result)
   if arg.dump_pdbs:
      result.dump_pdbs_top_score(hscore=hscore, **arg)
      result.dump_pdbs_top_score_each(hscore=hscore, **arg)
   if not arg.suppress_dump_results:
      rp.util.dump(result, arg.output_prefix + '_Result.pickle')

if __name__ == '__main__':
   main()