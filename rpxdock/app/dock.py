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

def gcd(a,b):
   while b > 0:
      a, b = b, a % b
   return a

## All dock_cyclic, dock_onecomp, and dock_multicomp do similar things
   
def dock_asym(hscore, **kw):
   kw = rp.Bunch(kw)
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
   
   sampler = rp.sampling.XformHier_f4(cartlb, cartub, cartbs, kw.ori_resl)
   logging.info(f'num base samples {sampler.size(0):,}')

   bodies = [[rp.Body(fn, allowed_res=ar2, **kw)
              for fn, ar2 in zip(inp, ar)]
             for inp, ar in zip(kw.inputs, kw.allowed_residues)]

   
   exe = concurrent.futures.ProcessPoolExecutor
   # exe = rp.util.InProcessExecutor
   with exe(kw.ncpu) as pool:
      futures = list()
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
      result = [None] * len(futures)
      for f in tqdm.tqdm(concurrent.futures.as_completed(futures), total=len(futures)):
         result[f.ijob] = f.result()
   result = rp.concat_results(result)
   return result

def dock_cyclic(hscore, **kw):
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
   kw = rp.Bunch(kw)
   spec = get_spec(kw.architecture)
   crtbnd = kw.cart_bounds[0]

   # If term_access or directions for termini given
   if (True in kw.term_access1) or (True in kw.termini_dir1) or (False in kw.termini_dir1):
      kw.poses = []
      kw.force_flip = [False] * len(kw.inputs)
      N_in = None
      C_in = None
      pose = rp.rosetta.get_pose(kw.inputs1[0], kw.posecache)
      og_seqlen = pose.size() #length or original pose before we modify it 
      if kw.termini_dir1[0] is not None: N_in = rp.rosetta.helix_trix.N_term_in(pose, kw.term_access1[0])
      elif kw.term_access1[0]: rp.rosetta.helix_trix.append_Nhelix(pose)
      if kw.termini_dir1[1] is not None: C_in = rp.rosetta.helix_trix.C_term_in(pose, kw.term_access1[1])
      elif kw.term_access1[1]: rp.rosetta.helix_trix.append_Chelix(pose)
      
      if sum(kw.term_access1) > 0: kw.poses.append(pose) #Only update pose if adding on helices to termini
      dir_possible, error_msg = rp.rosetta.helix_trix.limit_flip_update_pose(pose, N_in, C_in, 1, **kw)
      if not dir_possible: raise ValueError(error_msg)
      # for p in kw.poses:
      #    pose.dump_pdb(f"s./temp_dump/sample.pdb")
      #assert dir_possible, error_msg
   assert False
   # double normal resolution, cuz why not?
   if kw.docking_method == 'grid':
      flip=list(spec.flip_axis[:3])
      force_flip=False if kw.force_flip is None else kw.force_flip[0]
      # print("force flip: ", force_flip)
      # print("flip componenets: ", kw.flip_components)
      if not kw.flip_components[0]:
         flip = None
      sampler = rp.sampling.grid_sym_axis(
         cart=np.arange(crtbnd[0], crtbnd[1], kw.grid_resolution_cart_angstroms),
         ang=np.arange(0, 360 / spec.nfold, kw.grid_resolution_ori_degrees),
         axis=spec.axis,
         flip=flip,
         force_flip=force_flip
         )
      search = rp.grid_search
   else:
      if spec.type == 'mirrorlayer':
         sampler = rp.sampling.hier_mirror_lattice_sampler(spec, resl=10, angresl=10, **kw)
      else:
         sampler = rp.sampling.hier_axis_sampler(spec.nfold, lb=crtbnd[0], ub=crtbnd[1], resl=5,
                                                 angresl=5, axis=spec.axis, flipax=spec.flip_axis, **kw)
      search = rp.hier_search

   # pose info and axes that intersect. Use list of modified poses to make bodies if such list exists
   if kw.poses and len(kw.poses) > 0:
      # og_seqlen = 0 if sum(kw.term_access1) is 0 else False #default set to 0, will get updated in body
      bodies = [
         rp.Body(pose1, allowed_res=allowedres, modified_term=modterm, og_seqlen=og_seqlen, og_source=inp, **kw)
         for pose1, allowedres, modterm, inp in zip(kw.poses, kw.allowed_residues1, kw.term_access, kw.inputs1)
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
   kw = rp.Bunch(kw)
   spec = get_spec(kw.architecture)
   # Determine accessibility and flip restriction before we make sampler
   kw.poses = []
   kw.og = []
   rp.rosetta.helix_trix.init_termini(**kw)
   sampler = rp.sampling.hier_multi_axis_sampler(spec, **kw)
   logging.info(f'num base samples {sampler.size(0):,}')

   # Then use poses if needed 
   if len(kw.poses) > 0:
      # og = []
      # for i in range(len(kw.inputs)):
      #    og.append([False]) if sum(kw.term_access[i]) > 0 else og.append([True])
      # bodies = [[rp.Body(pose2, allowed_res=ar2,original=og2, og_source=inp2, **kw)
      #          for pose2, ar2, og2, inp2 in zip(pose1, ar, og1, inp)]
      #          for pose1, ar, og1, inp in zip(kw.poses, kw.allowed_residues, og, kw.inputs)]
      # og = [] #store original lengths if the pose was modified by appending something
      # for i in range(len(kw.inputs)):
      #    if sum(kw.term_access[i]) > 0: og.append([0])
      #    else:
      #       pose = rp.rosetta.get_pose(kw.inputs[i][0], kw.posecache)
      #       og.append([pose.size()]) #length or original pose before we modify it 
      bodies = [[rp.Body(pose2, allowed_res=ar2,og_seqlen=og2, **kw)
               for pose2, ar2, og2, in zip(pose1, ar, og1)]
               for pose1, ar, og1, in zip(kw.poses, kw.allowed_residues, kw.og)]
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
      sampler = rp.sampling.hier_axis_sampler(lb=crtbnd[0], ub=crtbnd[1], resl=10,
                                              angresl=10, **kw)
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
   kw = rp.Bunch(kw)
   spec = get_spec(kw.architecture)
   flip = list(spec.flip_axis)

   if kw.docking_method.lower() == 'hier':
       if not kw.flip_components[0]:
          flip[0] = None
       if spec.nfold[0] == spec.nfold[1]:
          sampler1 = rp.sampling.hier_axis_sampler(spec.nfold[0],  lb=0, ub=100, resl=5, angresl=5, axis=[0,0,1], flipax=flip[0])
       else:
          sampler1 = rp.sampling.hier_axis_sampler(spec.nfold[0]*spec.nfold[1]/gcd(spec.nfold[0],spec.nfold[1]),
                  lb=0, ub=100, resl=5, angresl=5, axis=[0,0,1], flipax=flip[0])
       sampler2 = rp.sampling.ZeroDHier([np.eye(4),rp.homog.hrot([1,0,0],180)])
       if len(kw.flip_components) is 1 and not kw.flip_components[0]:
          sampler2 = rp.sampling.ZeroDHier(np.eye(4))
       if len(kw.flip_components) is not 1 and not kw.flip_components[1]:
          sampler2 = rp.sampling.ZeroDHier(np.eye(4))
       sampler = rp.sampling.CompoundHier(sampler2, sampler1)
       logging.info(f'num base samples {sampler.size(0):,}')
       search = rp.hier_search

   elif kw.docking_method.lower() =='grid':
       flip[0]=list(spec.flip_axis[0,:3])
       if not kw.flip_components[0]:
          flip[0] = None
       if spec.nfold[0] == spec.nfold[1]:
          sampler1 = rp.sampling.grid_sym_axis(
            cart=np.arange(0, 100, kw.grid_resolution_cart_angstroms),
            ang=np.arange(0, 360 / spec.nfold[0], kw.grid_resolution_ori_degrees),
            axis=spec.axis[0],
            flip=flip[0]
            )
       else:
          sampler1 = rp.sampling.grid_sym_axis(
            cart=np.arange(0, 100, kw.grid_resolution_cart_angstroms),
            ang=np.arange(0, 360 / (spec.nfold[0]*spec.nfold[1]/gcd(spec.nfold[0],spec.nfold[1])), kw.grid_resolution_ori_degrees),
            axis=spec.axis[0],
            flip=flip[0]
            )
       assert sampler1.ndim == 3
       if (len(kw.flip_components)==1 and not kw.flip_components[0]) or (len(kw.flip_components)==2 and not kw.flip_components[1]):
          shape = (2,)*sampler1.shape
          sampler = np.zeros(shape=shape)
          sampler[0,]=sampler1
          sampler[1,]=np.eye(4)
       else:
          shape = (2,2*len(sampler1),4,4)
          sampler = np.zeros(shape=shape)
          sampler[0,:len(sampler1)] = sampler1
          sampler[0,len(sampler1):] = sampler1
          sampler[1,:len(sampler1)] = np.eye(4)
          sampler[1,len(sampler1):] = rp.homog.hrot([1,0,0],180)
       sampler = np.swapaxes(sampler,0,1)
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

def main():
   kw = get_rpxdock_args()
   rp.options.print_options(kw)
   print(f'{" RUNNING dock.py:main ":=^80}')

   logging.info(f'weights: {kw.wts}')

   hscore = rp.CachedProxy(rp.RpxHier(kw.hscore_files, **kw))
   arch = kw.architecture

   # TODO commit to master AK
   #sym, comp = arch.split('_')

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

   print(result)
   if kw.dump_pdbs:
      result.dump_pdbs_top_score(hscore=hscore, **kw)
      result.dump_pdbs_top_score_each(hscore=hscore, **kw)
   if not kw.suppress_dump_results:
      rp.util.dump(result, kw.output_prefix + '_Result.pickle')

if __name__ == '__main__':
   main()
