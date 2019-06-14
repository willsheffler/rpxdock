import _pickle, threading, os, sys
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
import numpy as np, xarray as xr, sicdock as sd
from sicdock.geom import symframes
from sicdock.sampling import RotCart1Hier_f4, grid_sym_axis
from sicdock.search import concat_results, make_plugs, plug_get_sample_hierarchy

def testarg():
   arg = sd.app.defaults()
   arg.wts = sd.Bunch(plug=1.0, hole=1.0, ncontact=0.1, rpx=1.0)
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
   sampler = sd.search.plug.____PLUG_TEST_SAMPLE_HIERARCHY____(plug, hole, hscore)
   result = make_plugs(plug, hole, hscore, sampler, **arg)
   ref = sd.data.get_test_data('test_plug_hier')
   sd.search.assert_results_close(result, ref)

def test_plug_olig_hier(hscore, body_c3_mono, hole):
   arg = testarg().sub(plug_fixed_olig=True)
   hsamp = RotCart1Hier_f4(-120, 120, 20, 0, 120, 12, [0, 0, 1])
   result = make_plugs(body_c3_mono, hole, hscore, hsamp, search=sd.hier_search, **arg)
   refh, _ = sd.data.get_test_data('test_plug_olig_grid_vs_hier')
   sd.search.assert_results_close(result, refh, 10)

def test_plug_olig_grid(hscore, body_c3_mono, hole):
   arg = testarg().sub(plug_fixed_olig=True)
   # should match hsamp resl4 grid
   gcart = np.linspace(-119.625, 119.625, 20 * 16)
   gang = np.linspace(0.3125, 119.6875, 12 * 16)
   xgrid = grid_sym_axis(gcart, gang)
   resultg = make_plugs(body_c3_mono, hole, hscore, xgrid, search=sd.grid_search, **arg)
   _, refg = sd.data.get_test_data('test_plug_olig_grid_vs_hier')
   sd.search.assert_results_close(resultg, refg, 10)

if __name__ == "__main__":
   # plug = sd.Body(sd.data.datadir + '/pdb/dhr64.pdb')
   # sd.dump(plug, sd.data.datadir + '/body/dhr64.pickle')

   plug = sd.data.body_dhr64()
   # body_c3_mono = sd.data.body_c3_mono()
   hole = sd.data.body_small_c3_hole()

   test_plug_hier(sd.data.small_hscore(), plug, hole)
   # test_plug_olig_hier(sd.data.small_hscore(), body_c3_mono, hole)
   # test_plug_olig_grid(sd.data.small_hscore(), body_c3_mono, hole)
