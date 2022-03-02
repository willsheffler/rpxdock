#! /home/sheffler/.conda/envs/rpxdock/bin/python

import logging, itertools, concurrent, tqdm, rpxdock as rp
import numpy as np
# Get rid of this later, just to test output

from rpxdock.tests.util import test_parse_flip
from rpxdock.tests.sampling import modified_xform_grid
#to import tests from util
# import sys
#sys.path.insert(0, 'home/jenstanisl/my_rpxdock_code/rpxdock/tests/util')
# print(sys.path)
#from test_parse_flip import *

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
   else:
      spec = rp.search.DockSpec2CompCage(arch)
   return spec

## All dock_cyclic, dock_onecomp, and dock_multicomp do similar things
#def dock_cyclic(hscore, **kw):
#   kw = rp.Bunch(kw)
#   bodies = [
#      rp.Body(inp, allowed_res=allowedres, **kw)
#      for inp, allowedres in zip(kw.inputs1, kw.allowed_residues1)
#   ]
#   exe = concurrent.futures.ProcessPoolExecutor
#   # exe = rp.util.InProcessExecutor
#   with exe(kw.ncpu) as pool:
#      futures = list()
#      # where the magic happens
#      for ijob, bod in enumerate(bodies):
#         futures.append(
#            pool.submit(
#               rp.search.make_cyclic,
#               bod,
#               kw.architecture.upper(),
#               hscore,
#               **kw,
#            ))
#         futures[-1].ijob = ijob
#      result = [None] * len(futures)
#      for f in tqdm.tqdm(concurrent.futures.as_completed(futures), total=len(futures)):
#         result[f.ijob] = f.result()
#   result = rp.concat_results(result)
#
#   # result = rp.search.make_cyclic(body, architecture.upper(), hscore, **kw)
#
#   return result

# Adding edits here to adjust for termini orientation
# testing with --architecture [arch] --inputs1 [path] --flip_components [T/F]
def dock_onecomp(hscore, **kw):
   kw = rp.Bunch(kw)
   spec = get_spec(kw.architecture)
   crtbnd = kw.cart_bounds[0]

   # Maybe make a function for pdb processing
   if (True in kw.term_access1) or (True in kw.termini_dir1) or (False in kw.termini_dir1):
      # pdbfile = kw.inputs1[0]
      # if isinstance(pdbfile, str):
      #    import rpxdock.rosetta.triggers_init as ros
      #    if kw.posecache:
      #       pose = ros.get_pose_cached(pdbfile)
      #    else:
      #       pose = ros.pose_from_file(pdbfile)
      #       ros.assign_secstruct(pose)
      # print("updated3")
      kw.poses = []
      pose = rp.rosetta.get_pose(kw.inputs1[0], kw.posecache)
      # kw.term = [False, True]
      # if True in kw.termini_dir1 or False in kw.termini_dir1:
      if True:
         #term_flip = rp.util.somefunct(kw.flip_components[0]) #Parse flip_components, return bool (T means flip)
         # term_flip = test_parse_flip.test_processing_body(bodies[0].pdbfile) #Parse flip_components, return bool (T means flip)
         # print("Is the C term pointed in?:", term_flip)
         
         # print(len(kw.inputs))
         # if len(kw.term_access) == 1: 
         #    kw.term_access = kw.term_access * len(kw.inputs) * 2 #One for each terminus, two per input

         # N_in = test_parse_flip.test_N_term_in(bodies[0].pdbfile, False)
         # N_in = test_parse_flip.test_N_term_in(bodies[0].pdbfile, True)
         # keep_N = kw.term_access[0]
         N_in = None
         if kw.termini_dir1[0] is not None:
            N_in = test_parse_flip.test_N_term_in(pose, kw.term_access1[0])
            # print("N in? ", N_in)
         elif kw.term_access1[0]:
            rp.rosetta.helix_trix.append_Nhelix(pose)

         # C_in = test_parse_flip.test_C_term_in(bodies[0].pdbfile, False)
         # C_in = test_parse_flip.test_C_term_in(bodies[0].pdbfile, True)
         # print(kw.term_access)
         # keep_C = kw.term_access[1]
         C_in = None
         if kw.termini_dir1[1] is not None:
            C_in = test_parse_flip.test_C_term_in(pose, kw.term_access1[1])
            # print("C in? ", C_in)
         elif kw.term_access1[1]:
            rp.rosetta.helix_trix.append_Chelix(pose)
         
         # print(N_in, C_in, " desired : ", kw.termini_dir1[0], kw.termini_dir1[1])
         dir_possible, error_msg = rp.rosetta.helix_trix.limit_flip_update_pose(pose, N_in, C_in, 1, **kw)
         assert dir_possible, error_msg

         #if not term_info: # Use this to check if we need to prevent flipping based on term dir and specfied desired dir
         # Put this elsewhere, it's messy 
         # if (kw.term[0] == kw.term[1] == None) or (kw.term[0] == N_in and kw.term[1] == C_in):
         #    print("in correct orientation already: ", N_in, kw.term[0], C_in, kw.term[1])
         #    flip = None
         # elif ((N_in != None and C_in != None) and 
         #    ((N_in != C_in and kw.term[0] == kw.term[1]) or
         #    (N_in == C_in and kw.term[0] != kw.term[1]))):
         #    print("Not possible")
         #    assert False
         # elif ((kw.term[0] != None and kw.term[0] != N_in) or 
         #    (kw.term[1] != None and kw.term[1] != C_in) or
         #    (kw.term[0] == (not C_in) and kw.term[1] == (not N_in) )): 
         #    #flip once
         #    print("need to flip once")
         #    if not kw.flip_components[0]:
         #       print("not allowed to flip -> desired orientation not possible")
         #       assert False
         #    else:
         #       #flip
         #       print("smth to flip or restrict sample space")
         #       import rpxdock.rosetta.triggers_init as ros
         #       rot_matrix = ros.rosetta.numeric.xyzMatrix_double_t().cols(-1,0,0,0,1,0,0,0,-1)
         #       v = ros.rosetta.numeric.xyzVector_double_t(0,0,0)
         #       # print(type(rot_matrix), rot_matrix, v, type(v))
         #       pose.apply_transform_Rx_plus_v(rot_matrix, v)
         #       # new_input = "/home/jenstanisl/my_rpxdock_code/rpxdock/tests/app/output/rotated.pdb"
         #       # pose.dump_pdb(new_input)
         #       flip = None
         #       kw.poses1 = [pose]
         #       # print("modified successfully")
         #       # mod_pdb = "/home/jenstanisl/my_rpxdock_code/rpxdock/tests/app/output/keepnone_access_flag.pdb"
         #       # pose.dump_pdb(mod_pdb)
         #       # print("dumped")
         #    kw.flip_components = [False]
         # Do stuff
      # else: #this means termini_dir is empty, so just add helices w/o checking direction
      #    if kw.term_access1[0]: rp.helix_trix.append_Nhelix(pose)
      #    if kw.term_access1[1]: rp.helix_trix.append_Chelix(pose)
      #    kw.poses.append(pose) # May need to update this variable later

   # double normal resolution, cuz why not?
   if kw.docking_method == 'grid':
      flip=list(spec.flip_axis[:3])
      # print(kw.flip_components)
      if not kw.flip_components[0]:
         flip = None
      
      # Probably can delete below info
      # if kw.term_orient != []:
      #    limit_sampler = rp.rosetta.helix_trix.orient(kw.inputs1[0], **kw):
      #    if limit_sampler:
      #       flip = None
      #    elif limit_sampler == None:
      #       assert False
   
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

   # print("poses list: ", kw.poses)
   # print("flip components ", kw.flip_components)

   # Jenna additions about allowed residues. Probably not using this, delete later
   # print("allowed residues: ", kw.allowed_residues1[0].static, kw.allowed_residues1[0].dynamic) #what is value when nothing is passed in?
   # print("checking allowed residues, not set to none")
   # print(kw.allowed_residues1)
   # kw.allowed_residues1 = [None] # No longer working without this line?
   # if kw.allowed_residues1 == [None]:
   #    print("entered if")
   #    kw.allowed_residues1 = [DefaultResidueSelector("1:142")]
   # print("allowed residues: ", kw.allowed_residues1)
   # print("allowed residues: ", kw.allowed_residues1[0].static, kw.allowed_residues1[0].dynamic) #what is value when nothing is passed in?
   # for i,j in kw.items(): print(i, " : ", j)
   # if kw.allowed_residues1[0] == None and kw.modifications1[0] != None:
   #    kw.allowed_residues = []

   # making bodies from poses, gotta remember to update allowed residues files...
   # does the allowed residues variable allow me to
   # print(kw.poses1, kw.allowed_residues1, kw.term_access1)
   if len(kw.poses) > 0:
      bodies = [
         rp.Body(inp, allowed_res=allowedres, modified_term=modterm, **kw)
         for inp, allowedres, modterm in zip(kw.poses, kw.allowed_residues1, kw.term_access)
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

#def dock_multicomp(hscore, **kw):
#   kw = rp.Bunch(kw)
#   spec = get_spec(kw.architecture)
#   sampler = rp.sampling.hier_multi_axis_sampler(spec, **kw)
#   logging.info(f'num base samples {sampler.size(0):,}')
#
#   bodies = [[rp.Body(fn, allowed_res=ar2, **kw)
#              for fn, ar2 in zip(inp, ar)]
#             for inp, ar in zip(kw.inputs, kw.allowed_residues)]
#   assert len(bodies) == spec.num_components
#
#   exe = concurrent.futures.ProcessPoolExecutor
#   # exe = rp.util.InProcessExecutor
#   with exe(kw.ncpu) as pool:
#      futures = list()
#      for ijob, bod in enumerate(itertools.product(*bodies)):
#         futures.append(
#            pool.submit(
#               rp.search.make_multicomp,
#               bod,
#               spec,
#               hscore,
#               rp.hier_search,
#               sampler,
#               **kw,
#            ))
#         futures[-1].ijob = ijob
#      result = [None] * len(futures)
#      for f in tqdm.tqdm(concurrent.futures.as_completed(futures), total=len(futures)):
#         result[f.ijob] = f.result()
#   result = rp.concat_results(result)
#   return result


def main():
   kw = get_rpxdock_args()
   # Options for testing
   # kw.docking_method="grid"
   
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
   elif len(arch) == 2 or (arch[0] == 'D' and arch[2] == '_'):
      result = dock_onecomp(hscore, **kw)
   elif arch.startswith('PLUG'):
      result = dock_plug(hscore, **kw)
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
