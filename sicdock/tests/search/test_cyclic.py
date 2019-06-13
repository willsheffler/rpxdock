from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
import _pickle, threading, os, argparse, sys, numpy as np
import sicdock as sd

def test_make_cyclic_hier(hscore, body):
   arg = sd.app.defaults()
   arg.wts = sd.Bunch(ncontact=0.1, rpx=0.0)
   arg.beam_size = 1e4
   arg.max_bb_redundancy = 2.0
   arg.nout_debug = 3
   arg.max_trim = 0
   arg.max_longaxis_dot_z = 0.5
   arg.executor = ThreadPoolExecutor(min(4, arg.ncpu / 2))
   result = sd.search.make_cyclic(body, "C3", hscore, search=sd.hier_search, **arg)
   # sd.dump(result, 'sicdock/data/testdata/test_make_cyclic_hier.pickle')
   ref = sd.data.get_test_data('test_make_cyclic_hier')
   sd.search.assert_results_close(result, ref)

def test_make_cyclic_grid(hscore, body):
   arg = sd.app.defaults()
   arg.wts = sd.Bunch(ncontact=0.1, rpx=0.0)
   arg.beam_size = 1e4
   arg.max_bb_redundancy = 2.0
   arg.nout_debug = 3
   arg.max_trim = 0
   arg.max_longaxis_dot_z = 0.5
   arg.executor = ThreadPoolExecutor(min(4, arg.ncpu / 2))
   result = sd.search.make_cyclic(body, "C3", hscore, search=sd.hier_search, **arg)
   # sd.dump(result, 'sicdock/data/testdata/test_make_cyclic_hier.pickle')
   ref = sd.data.get_test_data('test_make_cyclic_hier')
   sd.search.assert_results_close(result, ref)

if __name__ == "__main__":
   hscore = sd.data.small_hscore()
   body = sd.data.body_dhr14()
   test_make_cyclic_hier(hscore, body)
