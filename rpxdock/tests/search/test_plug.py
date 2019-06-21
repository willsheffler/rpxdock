import _pickle, threading, os, sys
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
import numpy as np, xarray as xr, rpxdock as rp
from rpxdock.geom import symframes
from rpxdock.sampling import RotCart1Hier_f4, grid_sym_axis
from rpxdock.search import concat_results, make_plugs, plug_get_sample_hierarchy

def testarg():
   arg = rp.app.defaults()
   arg.wts = rp.Bunch(plug=1.0, hole=1.0, ncontact=0.1, rpx=1.0)
   arg.beam_size = 1e4
   arg.max_bb_redundancy = 3.0
   arg.max_longaxis_dot_z = 0.5
   arg.executor = ThreadPoolExecutor(min(4, arg.ncpu / 2))
   arg.multi_iface_summary = np.min
   return arg

def test_plug_hier(hscore, plug, hole):
   arg = testarg()
   arg.max_trim = 200
   # arg.nout_debug = 3
   # arg.output_prefix = 'test_plug_hier'
   sampler = rp.search.plug.plug_test_hier_sampler(plug, hole, hscore)
   # sampler = rp.search.plug.plug_get_sample_hierarchy(plug, hole, hscore)
   result = make_plugs(plug, hole, hscore, rp.hier_search, sampler, **arg)
   # rp.dump(result, 'rpxdock/data/testdata/test_plug_hier.pickle')
   ref = rp.data.get_test_data('test_plug_hier')
   rp.search.assert_results_close(result, ref)

def test_plug_olig_hier(hscore, body_c3_mono, hole):
   arg = testarg().sub(plug_fixed_olig=True)
   hsamp = RotCart1Hier_f4(-120, 120, 20, 0, 120, 12, [0, 0, 1])
   result = make_plugs(body_c3_mono, hole, hscore, rp.hier_search, hsamp, **arg)
   # rp.dump(result, 'rpxdock/data/testdata/test_plug_olig_hier.pickle')
   ref = rp.data.get_test_data('test_plug_olig_hier')
   rp.search.assert_results_close(result, ref, 10)

def test_plug_olig_grid(hscore, body_c3_mono, hole):
   arg = testarg().sub(plug_fixed_olig=True)
   # should match hsamp resl4 grid
   gcart = np.linspace(-119.625, 119.625, 20 * 16)
   gang = np.linspace(0.3125, 119.6875, 12 * 16)
   xgrid = grid_sym_axis(gcart, gang)
   result = make_plugs(body_c3_mono, hole, hscore, rp.grid_search, xgrid, **arg)
   # rp.dump(result, 'rpxdock/data/testdata/test_plug_olig_grid.pickle')
   ref = rp.data.get_test_data('test_plug_olig_grid')
   rp.search.assert_results_close(result, ref, 10)

if __name__ == "__main__":
   # plug = rp.Body(rp.data.datadir + '/pdb/dhr64.pdb')
   # rp.dump(plug, rp.data.datadir + '/body/dhr64.pickle')
   # c3m = rp.Body(rp.data.datadir + '/pdb/test_c3_mono.pdb')
   # rp.dump(c3m, rp.data.datadir + '/body/test_c3_mono.pickle')
   # hole = rp.Body(rp.data.datadir + '/pdb/small_c3_hole.pdb', sym=3)
   # rp.dump(hole, rp.data.datadir + '/body/small_c3_hole.pickle')

   hole = rp.data.get_body('small_c3_hole_sym3')
   plug = rp.data.get_body('dhr64')
   body_c3_mono = rp.data.get_body('test_c3_mono')
   hscore = rp.data.small_hscore()

   test_plug_hier(hscore, plug, hole)
   test_plug_olig_hier(hscore, body_c3_mono, hole)
   test_plug_olig_grid(hscore, body_c3_mono, hole)
