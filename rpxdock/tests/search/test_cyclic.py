import concurrent, os, argparse, sys, numpy as np, rpxdock as rp, pytest
from rpxdock.search import grid_search
from willutil import Bunch

def get_arg():
   kw = rp.app.defaults()
   kw.wts = Bunch(ncontact=0.1, rpx=1.0)
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

   # result.dump_pdbs_top_score(3)
   # print('num results:', len(result))
   # assert 0

   # rp.dump(result, 'rpxdock/data/testdata/test_make_cyclic_hier.pickle')
   ref = rp.data.get_test_data('test_make_cyclic_hier')
   rp.search.assert_results_close(result, ref)

def test_make_cyclic_grid(hscore, body):
   kw = get_arg()
   kw.max_trim = 0
   kw.cart_resl = 3
   kw.ori_resl = 20
   result = rp.search.make_cyclic(body, "C3", hscore, grid_search, **kw)

   # result.dump_pdbs_top_score(3)

   # rp.dump(result, 'rpxdock/data/testdata/test_make_cyclic_grid.pickle')
   ref = rp.data.get_test_data('test_make_cyclic_grid')

   # print(result.scores[:10])
   # print(ref.scores[:10])
   # assert np.allclose(result.scores, ref.scores)
   # for i in range(len(result.xforms)):
   # print(i)
   # print(result.xforms[i])
   # print(ref.xforms[i])
   # assert np.allclose(result.xforms, ref.xforms, atol=1e-3)
   # print(len(result.xforms))
   # print(len(ref.xforms))
   # for k in result.data:
   # print(k)
   # assert np.allclose(result[k], ref[k], atol=1e-3)
   # print('OK?')
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
#    kw.wts = Bunch(ncontact=0.1, rpx=0.0)
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

def debug_marisa_dhr01():
   kw = rp.app.defaults()
   kw.wts = Bunch(ncontact=0.01, rpx=1.0)
   kw.beam_size = 1e4
   kw.max_bb_redundancy = 2.0
   kw.recenter_input = True
   # kw.max_longaxis_dot_z = 0.5
   # kw.debug = True
   # kw.nout_debug = 3
   kw.executor = concurrent.futures.ThreadPoolExecutor(min(4, kw.ncpu / 2))
   kw.max_trim = 0

   # hscore = rp.score.RpxHier('ailv_h', hscore_data_dir='/home/erinyang/hscore/')
   hscore = rp.data.small_hscore()

   body = rp.Body('/home/sheffler/debug/marisa/input/dhr01.pdb', **kw)
   result = rp.search.make_cyclic(body, "C2", hscore, **kw)
   # result.dump_pdbs_top_score(**kw.sub(nout_top=50, output_prefix='debug_marisa'))
   # print('num results:', len(result))
   result = rp.concat_results([result])
   rp.dump(result, 'result_test.pickle')

   assert 0

if __name__ == "__main__":
   #   a = np.array([
   #      [-0.57024276, 0.6282228, -0.52930105, 12.375],
   #      [-0.734342, -0.10104251, 0.67121726, 0.],
   #      [0.36819205, 0.7714446, 0.5189489, 0.],
   #      [0., 0., 0., 1.],
   #   ])
   #   b = np.array([
   #      [-0.57024276, 0.6282228, -0.52930105, 12.375],
   #      [0.734342, 0.10104233, -0.6712172, 0.],
   #      [-0.36819202, -0.77144456, -0.51894915, 0.],
   #      [0., 0., 0., 1.],
   #   ])
   #   print('0', rp.homog.axis_ang_cen_of(a))
   #   print('1', rp.homog.axis_ang_cen_of(b))
   #   print('2', rp.homog.axis_angle_of(a @ np.linalg.inv(b)))
   #   print('3', rp.homog.axis_angle_of(b @ np.linalg.inv(a)))
   #   print('4', rp.homog.axis_angle_of(np.linalg.inv(b) @ a))
   #   print('5', rp.homog.axis_angle_of(np.linalg.inv(a) @ b))
   #   print('why not rotated around x like in')
   #   assert 0
   hscore = rp.data.small_hscore()
   body = rp.data.get_body('DHR14')
   # test_make_dimer_hier(hscore, body)
   # test_make_cyclic_hier_trim(hscore, body)
   test_make_cyclic_grid(hscore, body)
   # debug_marisa_dhr01()
