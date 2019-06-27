import concurrent, os, argparse, sys, numpy as np, rpxdock as rp

def get_arg():
   arg = rp.app.defaults()
   arg.wts = rp.Bunch(ncontact=0.1, rpx=1.0)
   arg.beam_size = 1e4
   arg.max_bb_redundancy = 2.0
   arg.max_longaxis_dot_z = 0.5
   # arg.debug = True
   # arg.nout_debug = 3
   arg.executor = concurrent.futures.ThreadPoolExecutor(min(4, arg.ncpu / 2))
   return arg

def test_make_cyclic_hier(hscore, body):
   arg = get_arg()
   arg.max_trim = 0
   result = rp.search.make_cyclic(body, "C3", hscore, **arg)
   # rp.dump(result, 'rpxdock/data/testdata/test_make_cyclic_hier.pickle')
   ref = rp.data.get_test_data('test_make_cyclic_hier')
   rp.search.assert_results_close(result, ref)

def test_make_cyclic_hier_trim(hscore, body):
   arg = get_arg()
   arg.max_trim = 100
   # arg.output_prefix = 'test_make_cyclic_hier_trim'
   # arg.nout_debug = 3
   result = rp.search.make_cyclic(body, "C3", hscore, **arg)

   # print(result.reslb.data)
   # print(result.resub.data)

   # result = result.sel(model=result.resub != 319)
   # print(result.reslb)
   # print(result.resub)
   # result.dump_pdbs_top_score(10, output_prefix='trim')

   # rp.dump(result, 'rpxdock/data/testdata/test_make_cyclic_hier_trim.pickle')
   ref = rp.data.get_test_data('test_make_cyclic_hier_trim')
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
   hscore = rp.data.small_hscore()
   body = rp.data.get_body('DHR14')
   test_make_cyclic_hier(hscore, body)
   test_make_cyclic_hier_trim(hscore, body)
   # test_make_cyclic_grid(hscore, body)
