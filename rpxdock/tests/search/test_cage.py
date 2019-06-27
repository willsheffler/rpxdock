import concurrent, os, argparse, sys, numpy as np, rpxdock as rp

def get_arg():
   arg = rp.app.defaults()
   arg.wts = rp.Bunch(ncontact=0.3, rpx=1.0)
   arg.beam_size = 2e4
   arg.max_bb_redundancy = 3.0
   arg.max_delta_h = 9999
   arg.nout_debug = 0
   arg.nout_top = 0
   arg.nout_each = 0
   arg.score_only_ss = 'H'
   arg.max_trim = 0
   # arg.debug = True
   arg.executor = concurrent.futures.ThreadPoolExecutor(min(4, arg.ncpu / 2))
   return arg

def _test_cage_slide(hscore, body_cageA, body_cageB):
   arg = get_arg().sub(nout_each=3)
   spec = rp.search.DockSpec2CompCage('T33')
   rp.search.samples_2xCyclic_slide(spec)

def _test_cage_hier_no_trim(hscore, body_cageA, body_cageB):
   arg = get_arg()

   spec = rp.search.DockSpec2CompCage('T33')
   sampler = rp.search.hier_2axis_sampler(spec, 50, 59)
   result = rp.search.make_cage([body_cageA, body_cageB], spec, hscore, rp.hier_search, sampler,
                                **arg)

   # result = rp.concat_results(result)
   # result.dump_pdbs_top_score(hscore=hscore,
   # **arg.sub(nout_top=10, output_prefix='test_cage_hier_no_trim'))

   # rp.dump(result, 'rpxdock/data/testdata/test_cage_hier_no_trim.pickle')
   ref = rp.data.get_test_data('test_cage_hier_no_trim')
   # print(ref)
   rp.search.assert_results_close(result, ref)

def _test_cage_hier_trim(hscore, body_cageA_extended, body_cageB_extended):
   arg = get_arg().sub(nout_debug=0, max_trim=150)
   arg.wts.ncontact = 0
   arg.output_prefix = 'test_cage_hier_trim'
   # arg.executor = None
   # print('test_cage_hier_trim', body_cageA_extended.nres)

   spec = rp.search.DockSpec2CompCage('T33')
   sampler = rp.search.hier_2axis_sampler(spec, 40, 49)
   body_cageA_extended.trim_direction = "C"
   body_cageB_extended.trim_direction = "C"
   bodies = [body_cageA_extended, body_cageB_extended]
   result = rp.search.make_cage(bodies, spec, hscore, rp.hier_search, sampler, **arg)
   # rp.dump(result, 'rpxdock/data/testdata/test_cage_hier_trim.pickle')
   ref = rp.data.get_test_data('test_cage_hier_trim')
   rp.search.assert_results_close(result, ref)

   # result = rp.concat_results(result)
   # result = result.sel(model=result.resub[:, 0] != 219)
   # # result = result.sel(model=result.resub < 118)
   # print('lbA', result.reslb.data[:, 0])
   # print('lbB', result.reslb.data[:, 1])
   # print('ubA', result.resub.data[:, 0])
   # print('ubB', result.resub.data[:, 1])
   # print(result.scores.data)
   # result.dump_pdbs_top_score(hscore=hscore,
   #                            **arg.sub(nout_top=10, output_prefix="test_cage_hier_trim"))

if __name__ == '__main__':
   import logging
   logging.getLogger().setLevel(level='DEBUG')
   # body1 = rp.Body(rp.data.pdbdir + '/T33_dn2_asymA.pdb')
   # body2 = rp.Body(rp.data.pdbdir + '/T33_dn2_asymB.pdb')
   # rp.dump(body1, rp.data.bodydir + '/T33_dn2_asymA.pickle')
   # rp.dump(body2, rp.data.bodydir + '/T33_dn2_asymB.pickle')

   hscore = rp.data.small_hscore()

   # hscore = rp.HierScore('ilv_h', hscore_data_dir='/home/sheffler/data/rpx/hscore')

   # hscore = rp.HierScore('ilv_h/1000', hscore_data_dir='/home/sheffler/data/rpx/hscore')

   body1 = rp.data.get_body('T33_dn2_asymA')
   body2 = rp.data.get_body('T33_dn2_asymB')
   _test_cage_hier_no_trim(hscore, body1, body2)

   # body1 = rp.data.get_body('T33_dn2_asymA_extended')
   # body2 = rp.data.get_body('T33_dn2_asymB_extended')
   # _test_cage_hier_trim(hscore, body1, body2)

   # body1 = rp.Body('/home/sheffler/tmp/T33_dn2_asymA.pdb.gz')
   # body2 = rp.Body('/home/sheffler/tmp/T33_dn2_asymB.pdb.gz')
   # wts = rp.Bunch(ncontact=0.1, rpx=1.0)
   # evaluator = rp.search.cage.CageEvaluatorNoTrim([body1, body2],
   #                                                rp.search.DockSpec2CompCage('T33'), hscore,
   #                                                wts=wts)
   # print(evaluator(np.stack([np.eye(4), np.eye(4)]))[0])
