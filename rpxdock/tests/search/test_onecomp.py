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

def test_layer_1comp(hscore, bodyC6):
   #this is not functional
   kw = get_arg()
   kw.wts.ncontact = 0.01
   kw.beam_size = 10000
   kw.iface_summary = np.median
   kw.max_delta_h = 9e9
   #kw.executor = None
   kw.nout_debug = 1

   bodies = [bodyC6]
   spec = rp.search.DockSpec1CompLayer('P6_6')
   nfold = 6
   sampler = rp.sampling.hier_axis_sampler(nfold, -100, 100, resl=10, angresl=10,
                                                 flip_components=True)
   #bounds for the hier_multi_axis_sampler are critical for getting dock solutions - need big search space?
   #bounds set limits in x,y,z cartesian space. limits are same in each direction, and each term refers to bounds for each component

   result = rp.search.make_onecomp(bodies, spec, hscore, rp.hier_search, sampler, **kw)

   result.dump_pdbs_top_score(hscore=hscore,
    **kw.sub(nout_top=5, output_prefix='test_layer_1comp_test', output_asym_only=False))

   rp.dump(result, '/home/cnfries/PycharmProjects/rpxdock-cnfries/rpxdock/data/testdata/test_layer_1comp.pickle')
   ref = rp.data.get_test_data('test_layer_1comp')
   rp.search.assert_results_close(result, ref)

def main():
   hscore = rp.data.small_hscore()
   # hscore = rp.RpxHier('ilv_h/1000', hscore_data_dir='/home/sheffler/data/rpx/hscore')
   C2 = rp.data.get_body('C2_REFS10_1')
   C3 = rp.data.get_body('C3_1na0-1_1')

   # test_cage_hier_onecomp_notrim(hscore, C3)
   # test_cage_hier_D3_onecomp_notrim(hscore, C3)
   # test_cage_hier_D3_2_onecomp_notrim(hscore, C2)
   # _test_cage_hier_onecomp_trim(hscore, C3)
   # test_cage_grid_onecomp_notrim(hscore, C3)

   # test_deepesh_1comp_bug(hscore)

   C6 = rp.data.get_body('C6_3H22')
   test_layer_1comp(hscore, C2)

if __name__ == '__main__':
   main()
