import rpxdock as rp, concurrent, pytest, numpy as np

def test_cage_hier_onecomp_notrim(hscore, bodyC3):
   kw = rp.app.defaults()
   kw.wts = rp.Bunch(ncontact=0.01, rpx=1.0)
   kw.beam_size = 2e4
   kw.max_bb_redundancy = 2.0
   kw.max_delta_h = 9999
   kw.nout_debug = 0
   kw.nout_top = 0
   kw.nout_each = 0
   kw.score_only_ss = 'H'
   kw.max_trim = 0
   kw.flip_components = [True]

   # kw.debug = True
   kw.executor = concurrent.futures.ThreadPoolExecutor(min(4, kw.ncpu / 2))

   spec = rp.search.DockSpec1CompCage('T3')
   sampler = rp.sampling.hier_axis_sampler(spec.nfold, lb=0, ub=100, resl=10, angresl=10,
                                           axis=spec.axis, flipax=spec.flip_axis, **kw)
   result = rp.search.make_onecomp(bodyC3, spec, hscore, rp.hier_search, sampler, **kw)
   # print(result)
   # result.dump_pdbs_top_score(
   # hscore=hscore, **kw.sub(nout_top=10, output_prefix='test_cage_hier_onecomp_notrim'))

   # rp.dump(result, 'rpxdock/data/testdata/test_cage_hier_onecomp_notrim.pickle')
   ref = rp.data.get_test_data('test_cage_hier_onecomp_notrim')
   rp.search.assert_results_close(result, ref)

   # result.dump_pdbs_top_score(10)

def test_cage_hier_D3_onecomp_notrim(hscore, bodyC3):
   kw = rp.app.defaults()
   kw.wts = rp.Bunch(ncontact=0.01, rpx=1.0)
   kw.beam_size = 2e4
   kw.max_bb_redundancy = 2.0
   kw.max_delta_h = 9999
   kw.nout_debug = 0
   kw.nout_top = 0
   kw.nout_each = 0
   kw.score_only_ss = 'H'
   kw.max_trim = 0
   kw.flip_components = [True]
   # kw.debug = True
   kw.executor = concurrent.futures.ThreadPoolExecutor(min(4, kw.ncpu / 2))

   spec = rp.search.DockSpec1CompCage('D3_3')
   sampler = rp.sampling.hier_axis_sampler(spec.nfold, lb=0, ub=100, resl=10, angresl=10,
                                           axis=spec.axis, flipax=spec.flip_axis, **kw)
   result = rp.search.make_onecomp(bodyC3, spec, hscore, rp.hier_search, sampler, **kw)
   # print(result)
   # result.dump_pdbs_top_score(
   # hscore=hscore, **kw.sub(nout_top=10, output_prefix='test_cage_hier_D3_onecomp_notrim'))

   # rp.dump(result, 'rpxdock/data/testdata/test_cage_hier_D3_onecomp_notrim.pickle')
   ref = rp.data.get_test_data('test_cage_hier_D3_onecomp_notrim')
   rp.search.assert_results_close(result, ref)

   # result.dump_pdbs_top_score(10)

def test_cage_hier_D3_2_onecomp_notrim(hscore, bodyC2):
   kw = rp.app.defaults()
   kw.wts = rp.Bunch(ncontact=0.01, rpx=1.0)
   kw.beam_size = 2e4
   kw.max_bb_redundancy = 2.0
   kw.max_delta_h = 9999
   kw.nout_debug = 0
   kw.nout_top = 0
   kw.nout_each = 0
   kw.score_only_ss = 'H'
   kw.max_trim = 0
   kw.flip_components = [True]
   # kw.debug = True
   kw.executor = concurrent.futures.ThreadPoolExecutor(min(4, kw.ncpu / 2))

   spec = rp.search.DockSpec1CompCage('D3_2')
   sampler = rp.sampling.hier_axis_sampler(spec.nfold, lb=0, ub=100, resl=10, angresl=10,
                                           axis=spec.axis, flipax=spec.flip_axis, **kw)
   result = rp.search.make_onecomp(bodyC2, spec, hscore, rp.hier_search, sampler, **kw)
   # print(result)
   # result.dump_pdbs_top_score(
   # hscore=hscore, **kw.sub(nout_top=10, output_prefix='test_cage_hier_D3_2_onecomp_notrim'))

   # rp.dump(result, 'rpxdock/data/testdata/test_cage_hier_D3_2_onecomp_notrim.pickle')
   ref = rp.data.get_test_data('test_cage_hier_D3_2_onecomp_notrim')
   rp.search.assert_results_close(result, ref)

   # result.dump_pdbs_top_score(10)

@pytest.mark.skip
def test_cage_hier_onecomp_trim(hscore, bodyC3):
   kw = rp.app.defaults()
   kw.wts = rp.Bunch(ncontact=0.01, rpx=1.0)
   kw.beam_size = 2e4
   kw.max_bb_redundancy = 2.0
   kw.max_delta_h = 9999
   kw.nout_debug = 0
   kw.nout_top = 0
   kw.nout_each = 0
   kw.score_only_ss = 'H'
   kw.max_trim = 200
   kw.trim_direction = 'C'
   kw.flip_components = [True]
   # kw.debug = True
   # kw.executor = concurrent.futures.ThreadPoolExecutor(min(4, kw.ncpu / 2))

   spec = rp.search.DockSpec1CompCage('T3')
   sampler = rp.sampling.hier_axis_sampler(spec.nfold, lb=0, ub=200, resl=5, angresl=5,
                                           axis=spec.axis, flipax=spec.flip_axis, **kw)
   result = rp.search.make_onecomp(bodyC3, spec, hscore, rp.hier_search, sampler, **kw)
   print(result)
   result.dump_pdbs_top_score(10)

   # rp.dump(result, 'rpxdock/data/testdata/test_cage_hier_onecomp_trim.pickle')
   # ref = rp.data.get_test_data('test_cage_hier_onecomp_trim')
   # rp.search.assert_results_close(result, ref)

def test_cage_grid_onecomp_notrim(hscore, bodyC3):
   kw = rp.app.defaults()
   kw.wts = rp.Bunch(ncontact=0.01, rpx=1.0)
   kw.beam_size = 2e4
   kw.max_bb_redundancy = 2.0
   kw.max_delta_h = 9999
   kw.nout_debug = 0
   kw.nout_top = 0
   kw.nout_each = 0
   kw.score_only_ss = 'H'
   kw.max_trim = 0
   kw.flip_components = [True]

   # kw.debug = True
   kw.executor = concurrent.futures.ThreadPoolExecutor(min(4, kw.ncpu / 2))

   spec = rp.search.DockSpec1CompCage('T3')
   # sampler = rp.sampling.hier_axis_sampler(spec.nfold, lb=0, ub=100, resl=10, angresl=10,  axis=spec.axis,
   sampler = rp.sampling.grid_sym_axis(np.arange(0, 100, 2), np.arange(0, 360 / spec.nfold, 2),
                                       axis=spec.axis, flip=list(spec.flip_axis[:3]))

   result = rp.search.make_onecomp(bodyC3, spec, hscore, rp.grid_search, sampler, **kw)
   # print(result)
   # result.dump_pdbs_top_score(hscore=hscore,
   #                            **kw.sub(nout_top=3, output_prefix='test_cage_hier_onecomp_notrim'))
   #
   # rp.dump(result, 'rpxdock/data/testdata/test_cage_grid_onecomp_notrim.pickle')
   ref = rp.data.get_test_data('test_cage_grid_onecomp_notrim')
   rp.search.assert_results_close(result, ref)

   # result.dump_pdbs_top_score(10)
   # assert 0

def test_deepesh_1comp_bug(hscore):
   # print('start')
   # import sys
   # sys.stdout.flush()
   kw = rp.app.defaults()
   kw.wts = rp.Bunch(ncontact=0.01, rpx=1.0)
   kw.beam_size = 2e4
   kw.max_bb_redundancy = 2
   kw.max_delta_h = 9999
   kw.nout_debug = 0
   kw.nout_top = 0
   kw.nout_each = 0
   kw.score_only_ss = 'H'
   kw.max_trim = 0
   kw.flip_components = [True]

   # kw.debug = True
   # kw.executor = concurrent.futures.ThreadPoolExecutor(min(4, kw.ncpu / 2))

   spec = rp.search.DockSpec1CompCage('I3')
   body = rp.data.get_body('deepesh_1comp_bug')

   kw.sampler = rp.sampling.hier_axis_sampler(spec.nfold, lb=147, ub=154, resl=10, angresl=10,
                                              axis=spec.axis, flipax=spec.flip_axis, **kw)
   result = rp.search.make_onecomp(body, spec, hscore, **kw)

   # print(result)
   # result.dump_pdbs_top_score(
   #    hscore=hscore,
   #    **kw.sub(nout_top=3, output_prefix='test_cage_hier_onecomp_notrim'),
   # )
   # return

   # assert 0
   # rp.dump(result, 'rpxdock/data/testdata/test_deepesh_1comp_bug.pickle')
   ref = rp.data.get_test_data('test_deepesh_1comp_bug')
   rp.search.assert_results_close(result, ref)


def test_cage_onecomp_hier_termini_dirs(hscore, C3_1nza, bodyC3):
   poseC3 = C3_1nza
   C3_Ndir, C3_Cdir= True, False # init relative dirs of N and C terms
   dirs = [[[C3_Ndir, C3_Cdir]], [[not(C3_Ndir), not(C3_Cdir)]], [[None, None]]]
   expected_flips = [[False, False],[True, True], [True, False]]
   # expected vals of flip_components and force_flip after each trial

   result = [None] * len(dirs)
   for i, dir_pair in enumerate(dirs):
      kw = rp.app.defaults()
      kw.wts = rp.Bunch(ncontact=0.01, rpx=1.0)
      kw.beam_size = 2e4
      kw.max_bb_redundancy = 2.0
      kw.max_delta_h = 9999
      kw.nout_debug = 0
      kw.nout_top = 0
      kw.nout_each = 0
      kw.score_only_ss = 'H'
      kw.max_trim = 0
      kw.executor = concurrent.futures.ThreadPoolExecutor(min(4, kw.ncpu / 2))

      # Additions for termini direction control
      kw.inputs=[[poseC3]]
      kw.flip_components = [True]
      kw.force_flip = [False]
      kw.termini_dir = [dir_pair]
      kw.term_access=[[[False, False]]]
      poses, og_lens = rp.rosetta.helix_trix.init_termini(**kw) 

      assert [kw.flip_components[0], kw.force_flip[0]] == expected_flips[i]
      spec = rp.search.DockSpec1CompCage('T3')
      sampler = rp.sampling.hier_axis_sampler(spec.nfold, lb=0, ub=100, resl=10, angresl=10,
                                           axis=spec.axis, flipax=spec.flip_axis, **kw)
      result[i] = rp.search.make_onecomp(bodyC3, spec, hscore, rp.hier_search, sampler, **kw)
   
   assert not(result[0].__eq__(result[1]))
   assert not(result[1].__eq__(result[2]))
   assert not(result[0].__eq__(result[2]))
 
   result = rp.concat_results(result)
   # print(result)
   # result.dump_pdbs_top_score(hscore=hscore,
   #    **kw.sub(nout_top=10, use_body_sym=True, 
   #    output_prefix='/home/jenstanisl/test_rpx/unit_test_input/unit_test_temp_output/test_cage_onecomp_hier_termini_dirs'))

   # rp.dump(result, 'rpxdock/data/testdata/test_cage_onecomp_hier_termini_dirs.pickle')
   ref = rp.data.get_test_data('test_cage_onecomp_hier_termini_dirs') 
   rp.search.assert_results_close(result, ref)

def test_cage_onecomp_grid_termini_dirs(hscore, C3_1nza, bodyC3):
   poseC3 = C3_1nza
   C3_Ndir, C3_Cdir= True, False # init relative dirs of N and C terms
   dirs = [[[C3_Ndir, C3_Cdir]], [[not(C3_Ndir), not(C3_Cdir)]], [[None, None]]]
   expected_flips = [[False, False],[True, True], [True, False]]

   result = [None] * len(dirs)
   for i, dir_pair in enumerate(dirs):
      kw = rp.app.defaults()
      kw.wts = rp.Bunch(ncontact=0.01, rpx=1.0)
      kw.beam_size = 2e4
      kw.max_bb_redundancy = 2.0
      kw.max_delta_h = 9999
      kw.nout_debug = 0
      kw.nout_top = 0
      kw.nout_each = 0
      kw.score_only_ss = 'H'
      kw.max_trim = 0
      kw.executor = concurrent.futures.ThreadPoolExecutor(min(4, kw.ncpu / 2))

      # Additions for termini direction control
      kw.inputs=[[poseC3]]
      kw.flip_components = [True]
      kw.force_flip = [False]
      kw.termini_dir = [dir_pair]
      kw.term_access=[[[False, False]]]
      poses, og_lens = rp.rosetta.helix_trix.init_termini(**kw)
      force_flip=False if kw.force_flip is None else kw.force_flip[0]

      assert [kw.flip_components[0], kw.force_flip[0]] == expected_flips[i]
      spec = rp.search.DockSpec1CompCage('T3')
      sampler = rp.sampling.grid_sym_axis(np.arange(0, 100, 2), np.arange(0, 360 / spec.nfold, 2),
                                       axis=spec.axis, flip=list(spec.flip_axis[:3]), 
                                       force_flip=force_flip)
      result[i] = rp.search.make_onecomp(bodyC3, spec, hscore, rp.grid_search, sampler, **kw)

   assert not(result[0].__eq__(result[1]))
   assert not(result[1].__eq__(result[2]))
   assert not(result[1].__eq__(result[2]))

   result = rp.concat_results(result)
   # print(result)
   # result.dump_pdbs_top_score(hscore=hscore,
   #                            **kw.sub(nout_top=10, output_prefix='test_cage_onecomp_grid_termini_dirs'))

   # rp.dump(result, 'rpxdock/data/testdata/test_cage_onecomp_grid_termini_dirs.pickle')
   ref = rp.data.get_test_data('test_cage_onecomp_grid_termini_dirs') 
   rp.search.assert_results_close(result, ref)

def test_cage_onecomp_grid_term_access(hscore, term_mod_C3s):
   result = [None] * len(term_mod_C3s)
   for i, body in enumerate(term_mod_C3s):
      kw = rp.app.defaults()
      kw.wts = rp.Bunch(ncontact=0.01, rpx=1.0)
      kw.beam_size = 2e4
      kw.max_bb_redundancy = 2.0
      kw.max_delta_h = 9999
      kw.nout_debug = 0
      kw.nout_top = 0
      kw.nout_each = 0
      kw.score_only_ss = 'H'
      kw.max_trim = 0
      kw.executor = concurrent.futures.ThreadPoolExecutor(min(4, kw.ncpu / 2))

      if not hasattr(body, 'modified_term'): body.modified_term=[False, False]
      kw.term_access = [body.modified_term]
      # kw.term_access= [term_access[i]]
      kw.termini_dir=[[None, None]]
      force_flip = False

      spec = rp.search.DockSpec1CompCage('T3')
      sampler = rp.sampling.grid_sym_axis(np.arange(0, 100, 2), np.arange(0, 360 / spec.nfold, 2),
                                       axis=spec.axis, flip=list(spec.flip_axis[:3]), 
                                       force_flip=force_flip)
      result[i] = rp.search.make_onecomp(body, spec, hscore, rp.grid_search, sampler, **kw)

   assert not(result[0].__eq__(result[1]))
   assert not(result[1].__eq__(result[2]))
   assert not(result[0].__eq__(result[2]))

   # for j in range(len(result)):
   #    result[j].dump_pdbs_top_score(hscore=hscore,
   #       **kw.sub(nout_top=10, output_prefix=f'test_cage_onecomp_grid_term_access{j}'))
   result = rp.concat_results(result)
   # print(result)

   # rp.dump(result, 'rpxdock/data/testdata/test_cage_onecomp_grid_term_access.pickle')
   ref = rp.data.get_test_data('test_cage_onecomp_grid_term_access') 
   rp.search.assert_results_close(result, ref)

def test_cage_onecomp_hier_term_access(hscore, term_mod_C3s):
   result = [None] * len(term_mod_C3s)
   for i, body in enumerate(term_mod_C3s):
      kw = rp.app.defaults()
      kw.wts = rp.Bunch(ncontact=0.01, rpx=1.0)
      kw.beam_size = 2e4
      kw.max_bb_redundancy = 2.0
      kw.max_delta_h = 9999
      kw.nout_debug = 0
      kw.nout_top = 0
      kw.nout_each = 0
      kw.score_only_ss = 'H'
      kw.max_trim = 0
      kw.executor = concurrent.futures.ThreadPoolExecutor(min(4, kw.ncpu / 2))

      # kw.term_access= [term_access[i]]
      if not hasattr(body, 'modified_term'): body.modified_term=[False, False]
      kw.term_access = [body.modified_term]
      kw.termini_dir=[[None, None]]
      kw.force_flip = [False]
      kw.flip_components = [True]

      spec = rp.search.DockSpec1CompCage('T3')
      sampler = rp.sampling.hier_axis_sampler(spec.nfold, lb=0, ub=100, resl=10, angresl=10,
                                           axis=spec.axis, flipax=spec.flip_axis, **kw)
      result[i] = rp.search.make_onecomp(body, spec, hscore, rp.hier_search, sampler, **kw)

   assert not(result[0].__eq__(result[1]))
   assert not(result[1].__eq__(result[2]))
   assert not(result[0].__eq__(result[2]))

   # for j in range(len(result)):
   #    print(len(result[j]))
   #    result[j].dump_pdbs_top_score(hscore=hscore, 
   #       **kw.sub(nout_top=10, output_prefix=f'test_cage_onecomp_hier_term_access{j}'))
   result = rp.concat_results(result)
   # print(result)

   # rp.dump(result, 'rpxdock/data/testdata/test_cage_onecomp_hier_term_access.pickle')
   ref = rp.data.get_test_data('test_cage_onecomp_hier_term_access') 
   rp.search.assert_results_close(result, ref)

def main():
   hscore = rp.data.small_hscore()
   # hscore = rp.RpxHier('ilv_h/1000', hscore_data_dir='/home/sheffler/data/rpx/hscore')
   C2 = rp.data.get_body('C2_REFS10_1')
   C3 = rp.data.get_body('C3_1na0-1_1')

   test_cage_hier_onecomp_notrim(hscore, C3)
   test_cage_hier_D3_onecomp_notrim(hscore, C3)
   test_cage_hier_D3_2_onecomp_notrim(hscore, C2)
   # _test_cage_hier_onecomp_trim(hscore  , C3)
   test_cage_grid_onecomp_notrim(hscore, C3)
   # test_cage_hier_onecomp_notrim(hscore, C3)
   # test_cage_hier_D3_onecomp_notrim(hscore, C3)
   # test_cage_hier_D3_2_onecomp_notrim(hscore, C2)
   # _test_cage_hier_onecomp_trim(hscore, C3)
   # test_cage_grid_onecomp_notrim(hscore, C3)

   from rpxdock.rosetta.triggers_init import get_pose_cached
   poseC3 = get_pose_cached('C3_1na0-1_1.pdb.gz', rp.data.pdbdir)
   test_cage_onecomp_hier_termini_dirs(hscore, poseC3, C3)
   test_cage_onecomp_grid_termini_dirs(hscore, poseC3, C3)

   bodies = [rp.data.get_body('C3_3e6q_asu'),
            rp.data.get_body('C3_3e6q_asu_Chelix'),
            rp.data.get_body('C3_3e6q_asu_NChelix')]
   test_cage_onecomp_grid_term_access(hscore, bodies)
   test_cage_onecomp_hier_term_access(hscore, bodies)

   # test_deepesh_1comp_bug(hscore)
   

if __name__ == '__main__':
   main()
