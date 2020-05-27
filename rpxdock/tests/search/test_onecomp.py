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
   # kw.debug = True
   kw.executor = concurrent.futures.ThreadPoolExecutor(min(4, kw.ncpu / 2))

   spec = rp.search.DockSpec1CompCage('T3')
   sampler = rp.sampling.hier_axis_sampler(spec.nfold, lb=0, ub=100, axis=spec.axis,
                                           flipax=spec.flip_axis)
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
   # kw.debug = True
   kw.executor = concurrent.futures.ThreadPoolExecutor(min(4, kw.ncpu / 2))

   spec = rp.search.DockSpec1CompCage('D3_3')
   sampler = rp.sampling.hier_axis_sampler(spec.nfold, lb=0, ub=100, axis=spec.axis,
                                           flipax=spec.flip_axis)
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
   # kw.debug = True
   kw.executor = concurrent.futures.ThreadPoolExecutor(min(4, kw.ncpu / 2))

   spec = rp.search.DockSpec1CompCage('D3_2')
   sampler = rp.sampling.hier_axis_sampler(spec.nfold, lb=0, ub=100, axis=spec.axis,
                                           flipax=spec.flip_axis)
   result = rp.search.make_onecomp(bodyC2, spec, hscore, rp.hier_search, sampler, **kw)
   # print(result)
   # result.dump_pdbs_top_score(
   # hscore=hscore, **kw.sub(nout_top=10, output_prefix='test_cage_hier_D3_2_onecomp_notrim'))

   # rp.dump(result, 'rpxdock/data/testdata/test_cage_hier_D3_2_onecomp_notrim.pickle')
   ref = rp.data.get_test_data('test_cage_hier_D3_2_onecomp_notrim')
   rp.search.assert_results_close(result, ref)

   # result.dump_pdbs_top_score(10)

@pytest.mark.xfail
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
   # kw.debug = True
   # kw.executor = concurrent.futures.ThreadPoolExecutor(min(4, kw.ncpu / 2))

   spec = rp.search.DockSpec1CompCage('T3')
   sampler = rp.sampling.hier_axis_sampler(spec.nfold, lb=0, ub=200, resl=5, angresl=5,
                                           axis=spec.axis, flipax=spec.flip_axis)
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
   # kw.debug = True
   kw.executor = concurrent.futures.ThreadPoolExecutor(min(4, kw.ncpu / 2))

   spec = rp.search.DockSpec1CompCage('T3')
   # sampler = rp.sampling.hier_axis_sampler(spec.nfold, lb=0, ub=100, axis=spec.axis,
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

def main():
   hscore = rp.data.small_hscore()
   # hscore = rp.RpxHier('ilv_h/1000', hscore_data_dir='/home/sheffler/data/rpx/hscore')
   # C2 = rp.data.get_body('C2_REFS10_1')
   C3 = rp.data.get_body('C3_1na0-1_1')

   # test_cage_hier_onecomp_notrim(hscore, C3)
   # test_cage_hier_D3_onecomp_notrim(hscore, C3)
   # test_cage_hier_D3_2_onecomp_notrim(hscore, C2)
   # _test_cage_hier_onecomp_trim(hscore, C3)
   test_cage_grid_onecomp_notrim(hscore, C3)

if __name__ == '__main__':
   main()