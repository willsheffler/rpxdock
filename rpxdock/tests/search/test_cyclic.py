from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
import _pickle, threading, os, argparse, sys, numpy as np
import rpxdock as rp

def test_make_cyclic_hier(hscore, body):
   arg = rp.app.defaults()
   arg.wts = rp.Bunch(ncontact=0.1, rpx=0.0)
   arg.beam_size = 1e4
   arg.max_bb_redundancy = 2.0
   # arg.nout_debug = 3
   arg.max_trim = 0
   arg.max_longaxis_dot_z = 0.5
   arg.executor = ThreadPoolExecutor(min(4, arg.ncpu / 2))
   result = rp.search.make_cyclic(body, "C3", hscore, search=rp.hier_search, **arg)
   # rp.dump(result, 'rpxdock/data/testdata/test_make_cyclic_hier.pickle')
   ref = rp.data.get_test_data('test_make_cyclic_hier')
   rp.search.assert_results_close(result, ref)

# def test_make_cyclic_grid(hscore, body):
#    arg = rp.app.defaults()
#    arg.wts = rp.Bunch(ncontact=0.1, rpx=0.0)
#    arg.beam_size = 1e4
#    arg.max_bb_redundancy = 2.0
#    # arg.nout_debug = 3
#    arg.max_trim = 0
#    arg.max_longaxis_dot_z = 0.5
#    arg.executor = ThreadPoolExecutor(min(4, arg.ncpu / 2))
#    result = rp.search.make_cyclic(body, "C3", hscore, search=rp.hier_search, **arg)
#    # rp.dump(result, 'rpxdock/data/testdata/test_make_cyclic_grid.pickle')
#    ref = rp.data.get_test_data('test_make_cyclic_grid')
#    rp.search.assert_results_close(result, ref)

if __name__ == "__main__":
   # plug = rp.Body(rp.data.datadir + '/pdb/DHR14.pdb')
   # rp.dump(plug, rp.data.datadir + '/body/dhr14.pickle')
   hscore = rp.data.small_hscore()
   body = rp.data.body_dhr14()
   test_make_cyclic_hier(hscore, body)
   # test_make_cyclic_grid(hscore, body)
