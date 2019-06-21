import concurrent, os, argparse, sys, numpy as np, rpxdock as rp

def test_cage_slide(hscore, body_cageA, body_cageB):
   arg = rp.app.defaults()
   arg.wts = rp.Bunch(ncontact=0.3, rpx=1.0)
   arg.beam_size = 2e4
   arg.max_bb_redundancy = 3.0
   arg.max_trim = 0
   arg.max_delta_h = 9999
   arg.nout_debug = 0
   arg.nout_top = 0
   arg.nout_each = 3
   arg.max_trim = 100
   arg.score_only_ss = 'H'
   # arg.max_trim = 0
   # arg.executor = concurrent.futures.ThreadPoolExecutor(min(4, arg.ncpu / 2))

   spec = rp.search.DockSpec2CompCage('T33')
   rp.search.samples_2xCyclic_slide(spec)

def test_cage_hier(hscore, body_cageA, body_cageB):
   arg = rp.app.defaults()
   arg.wts = rp.Bunch(ncontact=0.3, rpx=1.0)
   arg.beam_size = 2e4
   arg.max_bb_redundancy = 3.0
   arg.max_trim = 0
   arg.max_delta_h = 9999
   arg.nout_debug = 0
   arg.nout_top = 0
   arg.nout_each = 3
   arg.max_trim = 100
   arg.score_only_ss = 'H'
   # arg.max_trim = 0
   # arg.executor = concurrent.futures.ThreadPoolExecutor(min(4, arg.ncpu / 2))

   spec = rp.search.DockSpec2CompCage('T33')
   sampler = rp.search.hier_cage_sampler(spec, 50, 100)

   result = rp.search.make_cage([body_cageA, body_cageB], spec, hscore, rp.hier_search, sampler,
                                **arg)
   print(np.max(result.scores))
   arg.max_trim = 0
   result2 = rp.search.make_cage([body_cageA, body_cageB], spec, hscore, rp.hier_search, sampler,
                                 **arg)
   print(np.max(result2.scores))

   result = rp.concat_results([result, result2])
   result.dump_pdbs_top_score_each(hscore=hscore, **arg)
   print(result)

if __name__ == '__main__':
   import logging
   logging.getLogger().setLevel(level='DEBUG')
   # body1 = rp.Body(rp.data.pdbdir + '/T33_dn2_asymA.pdb')
   # body2 = rp.Body(rp.data.pdbdir + '/T33_dn2_asymB.pdb')
   # rp.dump(body1, rp.data.bodydir + '/T33_dn2_asymA.pickle')
   # rp.dump(body2, rp.data.bodydir + '/T33_dn2_asymB.pickle')

   # hscore = rp.data.small_hscore()
   # hscore = rp.HierScore('ilv_h', hscore_data_dir='/home/sheffler/data/rpx/hscore')
   hscore = rp.HierScore('ilv_h/1000', hscore_data_dir='/home/sheffler/data/rpx/hscore')
   body1 = rp.data.get_body('T33_dn2_asymA')
   body2 = rp.data.get_body('T33_dn2_asymB')

   test_cage_hier(hscore, body1, body2)

   # body1 = rp.Body('/home/sheffler/tmp/T33_dn2_asymA.pdb.gz')
   # body2 = rp.Body('/home/sheffler/tmp/T33_dn2_asymB.pdb.gz')
   # wts = rp.Bunch(ncontact=0.1, rpx=1.0)
   # evaluator = rp.search.cage.CageEvaluatorNoTrim([body1, body2],
   #                                                rp.search.DockSpec2CompCage('T33'), hscore,
   #                                                wts=wts)
   # print(evaluator(np.stack([np.eye(4), np.eye(4)]))[0])
