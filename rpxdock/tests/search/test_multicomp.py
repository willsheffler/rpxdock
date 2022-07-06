import concurrent, os, argparse, sys, numpy as np, rpxdock as rp, pytest

def get_arg(**kw):
   arg = rp.app.defaults()
   arg.wts = rp.Bunch(ncontact=0.3, rpx=1.0)
   arg.beam_size = 2e4
   arg.max_bb_redundancy = 2.0
   arg.max_delta_h = 9999
   arg.nout_debug = 0
   arg.nout_top = 0
   arg.nout_each = 0
   arg.score_only_ss = 'H'
   arg.max_trim = 0
   # arg.debug = True
   arg.executor = concurrent.futures.ThreadPoolExecutor(min(4, arg.ncpu / 2))
   return arg.sub(kw)

def _test_cage_slide(hscore, body_cageA, body_cageB):
   # arg = get_arg().sub(nout_each=3)
   spec = rp.search.DockSpec2CompCage('T33')
   rp.search.samples_2xCyclic_slide(spec)

def test_cage_hier_no_trim(hscore, body_cageA, body_cageB):
   kw = get_arg(fixed_components=True)
   kw.beam_size = 5000

   spec = rp.search.DockSpec2CompCage('T33')
   sampler = rp.sampling.hier_multi_axis_sampler(spec, [50, 60])
   result = rp.search.make_multicomp([body_cageA, body_cageB], spec, hscore, rp.hier_search,
                                     sampler, **kw)
   # print(result)
   # result.dump_pdbs_top_score(hscore=hscore,
   #                            **kw.sub(nout_top=10, output_prefix='test_cage_hier_no_trim'))

   # rp.dump(result, 'rpxdock/data/testdata/test_cage_hier_no_trim.pickle')
   ref = rp.data.get_test_data('test_cage_hier_no_trim')
   rp.search.assert_results_close(result, ref)

def test_cage_hier_trim(hscore, body_cageA_extended, body_cageB_extended):
   kw = get_arg().sub(nout_debug=0, max_trim=150, fixed_components=True)
   kw.output_prefix = 'test_cage_hier_trim'
   kw.wts.ncontact = 0.0
   kw.trimmable_components = 'AB'
   kw.beam_size = 2000
   # kw.executor = None
   # print('test_cage_hier_trim', body_cageA_extended.nres)

   spec = rp.search.DockSpec2CompCage('T33')
   sampler = rp.sampling.hier_multi_axis_sampler(spec, [40, 80])
   body_cageA_extended.trim_direction = "C"
   body_cageB_extended.trim_direction = "C"
   bodies = [body_cageA_extended, body_cageB_extended]
   result = rp.search.make_multicomp(bodies, spec, hscore, rp.hier_search, sampler, **kw)

   # print(result)
   # result = result.sel(model=result.resub[:, 0] != 219)
   # # result = result.sel(model=result.resub < 118)
   # print('lbA', result.reslb.data[:, 0])
   # print('lbB', result.reslb.data[:, 1])
   # print('ubA', result.resub.data[:, 0])
   # print('ubB', result.resub.data[:, 1])
   # print(result.scores.data)
   # result.dump_pdbs_top_score(hscore=hscore,
   # **kw.sub(nout_top=10, output_prefix="test_cage_hier_trim"))
   # result.resub[:] = np.max(result.resub, axis=0)
   # result.dump_pdbs_top_score(hscore=hscore,
   # **kw.sub(nout_top=10, output_prefix="whole_test_cage_hier_trim"))

   # rp.dump(result, 'rpxdock/data/testdata/test_cage_hier_trim.pickle')
   ref = rp.data.get_test_data('test_cage_hier_trim')
   rp.search.assert_results_close(result, ref)

def test_cage_hier_3comp(hscore, bodyC4, bodyC3, bodyC2):
   kw = get_arg()
   kw.wts.ncontact = 0.01
   kw.beam_size = 10000
   kw.iface_summary = np.median

   bodies = [bodyC4, bodyC3, bodyC2]
   spec = rp.search.DockSpec3CompCage('O432')
   sampler = rp.sampling.hier_multi_axis_sampler(spec, [70, 90], flip_components=False)
   result = rp.search.make_multicomp(bodies, spec, hscore, rp.hier_search, sampler, **kw)

   # result.dump_pdbs_top_score(hscore=hscore,
   #                            **kw.sub(nout_top=10, output_prefix='test_cage_hier_3comp'))

   # rp.dump(result, 'rpxdock/data/testdata/test_cage_hier_3comp.pickle')
   ref = rp.data.get_test_data('test_cage_hier_3comp')
   rp.search.assert_results_close(result, ref)

@pytest.mark.skip
def test_layer_hier_3comp(hscore, bodyC6, bodyC3, bodyC2):
   kw = get_arg()
   kw.wts.ncontact = 0.01
   kw.beam_size = 10000
   kw.iface_summary = np.median
   kw.max_delta_h = 9e9
   # kw.executor = None
   # kw.nout_debug = 3

   bodies = [bodyC6, bodyC3, bodyC2]
   spec = rp.search.DockSpec3CompLayer('P6_632')
   sampler = rp.sampling.hier_multi_axis_sampler(spec, [[50, 100], [-10, 20], [-10, 20]],
                                                 flip_components=False)
   # sampler = rp.sampling.hier_multi_axis_sampler(spec, [[0, 300], [-10, 10], [-10, 10]],
   # flip_components=False)
   result = rp.search.make_multicomp(bodies, spec, hscore, rp.hier_search, sampler, **kw)

   # result.dump_pdbs_top_score(hscore=hscore,
   # **kw.sub(nout_top=10, output_prefix='test_layer_hier_3comp'))

   # rp.dump(result, 'rpxdock/data/testdata/test_layer_hier_3comp.pickle')
   ref = rp.data.get_test_data('test_layer_hier_3comp')
   rp.search.assert_results_close(result, ref)

# Check that specifying termini direction is restricting search properly
# Takes about 130 seconds
def test_cage_termini_dirs(hscore, bodyC3, bodyC2, poseC3, poseC2):
   import time
   start = time.time()

   C3_Ndir, C3_Cdir= True, False # init relative dirs of N and C terms
   C2_Ndir, C2_Cdir = True, False 
   dirs = [[[C3_Ndir, C3_Cdir],[C2_Ndir, C2_Cdir]],
            [[not(C3_Ndir), not(C3_Cdir)],[not(C2_Ndir), not(C2_Cdir)]],
            [[not(C3_Ndir), not(C3_Cdir)],[C2_Ndir, C2_Cdir]],
            [[C3_Ndir, C3_Cdir],[not(C2_Ndir), not(C2_Cdir)]],
            [[None,None], [None, None]]] 
   expected_flips = [[[False,False],[False,False]],
                     [[True,True],[True,True]],
                     [[True,True],[False,False]],
                     [[False,False],[True,True]],
                     [[True,False],[True,False]]]

   result = [None] * len(dirs)
   for i, dir_pair in enumerate(dirs):
      kw = get_arg()
      kw.beam_size = 5000
      kw.inputs = [[poseC3], [poseC2]]
      kw.flip_components = [True, True]
      kw.force_flip = [False,False]
      kw.term_access=[[[False, False]],[[False, False]]]
      kw.termini_dir = [[dir_pair[0]], [dir_pair[1]]]
      poses, og_lens = rp.rosetta.helix_trix.init_termini(**kw) 
      assert [kw.flip_components[0], kw.force_flip[0]] == expected_flips[i][0]
      assert [kw.flip_components[1], kw.force_flip[1]] == expected_flips[i][1]

      
      spec = rp.search.DockSpec2CompCage('I32')
      sampler = rp.sampling.hier_multi_axis_sampler(spec, **kw) 

      result[i] = rp.search.make_multicomp([bodyC3, bodyC2], spec, hscore, rp.hier_search,
                                       sampler, **kw)
   for j in range(len(result)):
      for k in range(j+1, len(result)):assert not(result[j].__eq__(result[k]))
   result = rp.concat_results(result)
   # print(result)
   # result.dump_pdbs_top_score(hscore=hscore,
   #                            **kw.sub(nout_top=10, output_prefix='test_cage_hier_no_trim'))

   # rp.dump(result, 'rpxdock/data/testdata/test_cage_termini_dirs.pickle')
   ref = rp.data.get_test_data('test_cage_termini_dirs') 
   rp.search.assert_results_close(result, ref)
   end = time.time()
   print("time to run ", end - start, "s")

# Check that various term_access parameters create different results
# Takes about 190 seconds
def test_cage_term_access(hscore, bodyC3, bodyC2):
   # import time
   # start = time.time()
   both_bodyC2 = rp.data.get_body('C2_REFS10_1_NChelix')
   N_bodyC2 = rp.data.get_body('C2_REFS10_1_Nhelix')
   both_bodyC3 = rp.data.get_body('C3_1na0-1_1_NChelix')
   C_bodyC3 = rp.data.get_body('C3_1na0-1_1_Chelix')
   # body_list = [[bodyC3, bodyC2], [both_bodyC3, bodyC2],
   #             [bodyC3, both_bodyC2], [both_bodyC3, both_bodyC2],
   #             [C_bodyC3, N_bodyC2]]
   body_list = [[bodyC3, bodyC2], [both_bodyC3, both_bodyC2],
               [C_bodyC3, N_bodyC2]]

   result = [None] * len(body_list)
   for i, bodies in enumerate(body_list):
      kw = get_arg()
      kw.beam_size = 5000
      kw.inputs = [[poseC3], [poseC2]]
      kw.flip_components = [True, True]
      kw.force_flip = [False,False]
      for comp in bodies:
         if not hasattr(comp, 'modified_term'):comp.modified_term=[False, False]
         kw.term_access.append([comp.modified_term])
      kw.termini_dir = [[[None]], [None]]
      assert len(kw.term_access) == len(kw.termini_dir)

      
      spec = rp.search.DockSpec2CompCage('I32')
      sampler = rp.sampling.hier_multi_axis_sampler(spec, **kw) 

      result[i] = rp.search.make_multicomp([bodies[0], bodies[1]], spec, hscore, rp.hier_search,
                                       sampler, **kw)
   for j in range(len(result)):
      for k in range(j+1, len(result)):assert not(result[j].__eq__(result[k]))
   result = rp.concat_results(result)
   # print(result)
   # result.dump_pdbs_top_score(hscore=hscore,
   #                            **kw.sub(nout_top=10, output_prefix='test_cage_hier_no_trim'))

   # rp.dump(result, 'rpxdock/data/testdata/test_cage_term_access.pickle')
   ref = rp.data.get_test_data('test_cage_term_access') 
   rp.search.assert_results_close(result, ref)
   # end = time.time()
   # print("time to run ", end - start, "s")

if __name__ == '__main__':
   import logging
   logging.getLogger().setLevel(level='INFO')
   logging.info('set loglevel to INFO')
   # body1 = rp.Body(rp.data.pdbdir + '/T33_dn2_asymA.pdb')
   # body2 = rp.Body(rp.data.pdbdir + '/T33_dn2_asymB.pdb')
   # rp.dump(body1, rp.data.bodydir + '/T33_dn2_asymA.pickle')
   # rp.dump(body2, rp.data.bodydir + '/T33_dn2_asymB.pickle')

   hscore = rp.data.small_hscore()
   # hscore = rp.RpxHier('ilv_h', hscore_data_dir='/home/sheffler/data/rpx/hscore')
   # hscore = rp.RpxHier('ilv_h/1000', hscore_data_dir='/home/sheffler/data/rpx/hscore')

   # body1 = rp.data.get_body('T33_dn2_asymA')
   # body2 = rp.data.get_body('T33_dn2_asymB')
   # test_cage_hier_no_trim(hscore, body1, body2)

   # body1 = rp.data.get_body('T33_dn2_asymA_extended')
   # body2 = rp.data.get_body('T33_dn2_asymB_extended')
   # test_cage_hier_trim(hscore, body1, body2)

   # C2 = rp.data.get_body('C2_REFS10_1')
   # C3 = rp.data.get_body('C3_1na0-1_1')
   # C4 = rp.data.get_body('C4_1na0-G1_1')
   # test_cage_hier_3comp(hscore, C4, C3, C2)

   # C2 = rp.data.get_body('C2_REFS10_1')
   # C3 = rp.data.get_body('C3_1na0-1_1')
   # C4 = rp.data.get_body('C4_1na0-G1_1')
   # C6 = rp.data.get_body('C6_3H22')
   # test_layer_hier_3comp(hscore, C6, C3, C2)

   C2 = rp.data.get_body('C2_REFS10_1')
   C3 = rp.data.get_body('C3_1na0-1_1')
   from rpxdock.rosetta.triggers_init import get_pose_cached
   poseC2 = get_pose_cached('C2_REFS10_1.pdb.gz', rp.data.pdbdir)
   poseC3 = get_pose_cached('C3_1na0-1_1.pdb.gz', rp.data.pdbdir)

   # test_cage_termini_dirs(hscore, C3, C2, poseC3, poseC2)
   test_cage_term_access(hscore, C2, C3)
