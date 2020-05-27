import concurrent, os, argparse, sys, numpy as np, rpxdock as rp

def get_arg():
   kw = rp.app.defaults()
   kw.wts = rp.Bunch(ncontact=0.1, rpx=1.0)
   kw.beam_size = 1e4
   kw.max_bb_redundancy = 2.0
   kw.max_longaxis_dot_z = 0.5
   # kw.debug = True
   # kw.nout_debug = 3
   kw.executor = concurrent.futures.ThreadPoolExecutor(min(4, kw.ncpu / 2))
   return kw

def test_make_cyclic_hier(hscore, body):
   kw = get_arg()
   kw.max_trim = 0
   result = rp.search.make_cyclic(body, "C3", hscore, **kw)
   # rp.dump(result, 'rpxdock/data/testdata/test_make_cyclic_hier.pickle')
   ref = rp.data.get_test_data('test_make_cyclic_hier')
   rp.search.assert_results_close(result, ref)

def test_make_cyclic_hier_trim(hscore, body):
   kw = get_arg()
   kw.max_trim = 100
   # kw.output_prefix = 'test_make_cyclic_hier_trim'
   # kw.nout_debug = 3
   result = rp.search.make_cyclic(body, "C3", hscore, **kw)

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
#    kw = rp.app.defaults()
#    kw.wts = rp.Bunch(ncontact=0.1, rpx=0.0)
#    kw.beam_size = 1e4
#    kw.max_bb_redundancy = 2.0
#    # kw.nout_debug = 3
#    kw.max_trim = 0
#    kw.max_longaxis_dot_z = 0.5
#    kw.executor = ThreadPoolExecutor(min(4, kw.ncpu / 2))
#    result = rp.search.make_cyclic(body, "C3", hscore, search=rp.hier_search, **kw)
#    # rp.dump(result, 'rpxdock/data/testdata/test_make_cyclic_grid.pickle')
#    ref = rp.data.get_test_data('test_make_cyclic_grid')
#    rp.search.assert_results_close(result, ref)

if __name__ == "__main__":
   hscore = rp.data.small_hscore()
   body = rp.data.get_body('DHR14')
   test_make_cyclic_hier(hscore, body)
   test_make_cyclic_hier_trim(hscore, body)
   # test_make_cyclic_grid(hscore, body)
