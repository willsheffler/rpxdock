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

   C2 = rp.data.get_body('C2_REFS10_1')
   C3 = rp.data.get_body('C3_1na0-1_1')
   C4 = rp.data.get_body('C4_1na0-G1_1')
   test_cage_hier_3comp(hscore, C4, C3, C2)

   # C2 = rp.data.get_body('C2_REFS10_1')
   # C3 = rp.data.get_body('C3_1na0-1_1')
   # C4 = rp.data.get_body('C4_1na0-G1_1')
   # C6 = rp.data.get_body('C6_3H22')
   # test_layer_hier_3comp(hscore, C6, C3, C2)
