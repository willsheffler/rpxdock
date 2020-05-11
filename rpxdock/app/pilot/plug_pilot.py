from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
import _pickle, threading, os, sys
from time import perf_counter
import numpy as np
import xarray as xr
import rpxdock
from rpxdock.motif import RpxHier
from rpxdock.util import GLOBALCACHE as HC
from rpxdock.data import datadir
from rpxdock.util import load, Bunch, load_threads
from rpxdock.body import Body
from rpxdock.io.io_body import dump_pdb_from_bodies
from rpxdock.geom import symframes
from rpxdock.sampling import RotCart1Hier_f4, grid_sym_axis
from rpxdock.search import concat_results, make_plugs, plug_get_sample_hierarchy
from rpxdock.search import grid_search
from rpxdock.search.plug import __make_plugs_hier_sample_test__, plug_test_hier_sampler

def quick_test_olig(hscore, cli_args=dict()):
   args = rpxdock.app.defaults()
   args.nresl = 5
   args.wts = Bunch(plug=1.0, hole=1.0, ncontact=0.1, rpx=1.0)
   args.beam_size = 1e4
   args.max_bb_redundancy = 3.0
   args.max_longaxis_dot_z = 0.5
   args.executor = ThreadPoolExecutor(args.ncpu)
   args.multi_iface_summary = np.min  # min(plug, hole)
   args.plug_fixed_olig = True
   # args.max_longaxis_dot_z = 0.1
   # args.multi_iface_summary = np.sum  # sum(plug, hole)
   # args.multi_iface_summary = lambda x, **kw: x[:, 0]  # plug only
   # args.wts = Bunch(plug=1.0, hole=1.0, ncontact=1.0, rpx=0)  # ncontact only

   args.sub(cli_args)

   cache = "/home/sheffler/debug/plug_from_oligomer/plug_from_olig.pickle"
   fplug = '/home/sheffler/debug/plug_from_oligomer/input/test_c3_mono.pdb'
   fhole = '/home/sheffler/debug/plug_from_oligomer/input/C3_o42_Z_asym.pdb'
   try:
      with open(cache, "rb") as inp:
         plug, hole = _pickle.load(inp)
   except:
      plug = Body(fplug)
      hole = Body(fhole, sym=3)
      with open(cache, "wb") as out:
         _pickle.dump([plug, hole], out)
   if args.nout_debug: dump_pdb_from_bodies("test_hole.pdb", [hole], symframes(hole.sym))

   hsamp = RotCart1Hier_f4(-120, 120, 20, 0, 120, 12, [0, 0, 1])
   result1 = make_plugs(plug, hole, hscore, hsamp, **kws)

   # should match hsamp resl4 grid
   gcart = np.linspace(-119.625, 119.625, 20 * 16)
   gang = np.linspace(0.3125, 119.6875, 12 * 16)
   xgrid = grid_sym_axis(gcart, gang)
   result2 = make_plugs(plug, hole, hscore, xgrid, search=grid_search, **kws)

   assert np.allclose(result1.scores[0], result2.scores[0])

def quick_test(hscore, cli_args=dict()):
   args = rpxdock.app.defaults()
   args.nout_debug = 3
   args.nresl = 5
   args.wts = Bunch(plug=1.0, hole=1.0, ncontact=0.1, rpx=1.0)
   args.beam_size = 1e4
   args.max_bb_redundancy = 3.0
   args.max_longaxis_dot_z = 0.5
   args.executor = ThreadPoolExecutor(args.ncpu)
   args.multi_iface_summary = np.min  # min(plug, hole)
   # args.max_longaxis_dot_z = 0.1
   # args.multi_iface_summary = np.sum  # sum(plug, hole)
   # args.multi_iface_summary = lambda x, **kw: x[:, 0]  # plug only
   # args.wts = Bunch(plug=1.0, hole=1.0, ncontact=1.0, rpx=0)  # ncontact only
   # args.output_prefix = "rpx"
   args.sub(cli_args)

   make_sampler = plug_test_hier_sampler
   try:
      with open("/home/sheffler/debug/rpxdock/hole.pickle", "rb") as inp:
         plug, hole = _pickle.load(inp)
      # plug = Body("/home/sheffler/scaffolds/repeat/dhr8/DHR05.pdb")
   except:
      plug = Body(datadir + "/pdb/DHR14.pdb")
      # hole = Body(datadir + "/pdb/hole_C3_tiny.pdb", sym=3)
      hole = HC(Body, "/home/sheffler/scaffolds/holes/C3_i52_Z_asym.pdb", sym=3)
      with open("/home/sheffler/debug/rpxdock/hole.pickle", "wb") as out:
         _pickle.dump([plug, hole], out)
   dump_pdb_from_bodies("test_hole.pdb", [hole], symframes(hole.sym))
   sampler = make_sampler(plug, hole, hscore)
   result = make_plugs(plug, hole, hscore, sampler, **kws)
   print(result)

def multiprocess_helper(ihole, iplug, make_sampler, args):
   global ___GLOBALS_MULTIPROCESS_TEST
   hscore = ___GLOBALS_MULTIPROCESS_TEST.hscore
   htag, hole = ___GLOBALS_MULTIPROCESS_TEST.holes[ihole]
   ptag, plug = ___GLOBALS_MULTIPROCESS_TEST.plugs[iplug]
   tag = f"{htag}_{ptag}"
   sampler = make_sampler(plug, hole, hscore)
   if args.nthread > 1:
      args.executor = ThreadPoolExecutor(args.nthread)
   result = make_plugs(plug, hole, hscore, sampler, output_prefix=tag, **kws)
   result.attrs['ihole'] = ihole
   result.attrs['iplug'] = iplug
   if args.executor:
      args.executor.shutdown(wait=False)
   return result

def load_body(f):
   tag = os.path.basename(f).replace(".pdb", "")
   if tag[0] == "C" and tag[2] == "_":
      body = Body(f, sym=tag[:2])
   else:
      body = Body(f)
   return tag, body

def threaded_load_hscore_and_bodies(hscore_files, body_files, nthread):
   loader = load_threads(hscore_files)
   loader.start()
   with ProcessPoolExecutor(nthread) as exe:
      bodies = list(exe.map(load_body, body_files))
   loader.join()
   hscore = RpxHier(loader.result)
   return hscore, bodies

def multiprocess_test(cli_args):
   args = rpxdock.app.defaults()
   args.nout_debug = 1
   args.nresl = 5
   args.wts = Bunch(plug=1.0, hole=1.0, ncontact=0.1, rpx=1.0)
   args.beam_size = 1e4
   args.max_bb_redundancy = 3.0
   args.max_longaxis_dot_z = 0.5
   args.multi_iface_summary = np.min  # min(plug, hole)
   args.trial_run = True
   d = "/home/sheffler/scaffolds/repeat/dhr8/"
   p = ["DHR01", "DHR03", "DHR04", "DHR05", "DHR07", "DHR08", "DHR09", "DHR10"]
   args.plugs = [d + x + ".pdb" for x in p]
   args.holes = [
      "/home/sheffler/scaffolds/holes/C3_i52_Z_asym.pdb",
      "/home/sheffler/scaffolds/holes/C3_o42_Z_asym.pdb",
   ]
   args = args.sub(cli_args)

   if len(args.hscore_files) is 1 and args.hscore_files[0] in globals():
      args.hscore_files = globals()[args.hscore_files[0]]
   hscore, bodies = threaded_load_hscore_and_bodies(args.hscore_files, args.holes + args.plugs,
                                                    args.nprocess * args.nthread)
   holes = bodies[:len(args.holes)]
   for htag, hole in holes:
      htag = args.output_prefix + htag if args.output_prefix else htag
      dump_pdb_from_bodies(htag + ".pdb", [hole], symframes(hole.sym))
   plugs = bodies[len(args.holes):]

   global ___GLOBALS_MULTIPROCESS_TEST
   ___GLOBALS_MULTIPROCESS_TEST = Bunch()
   ___GLOBALS_MULTIPROCESS_TEST.hscore = hscore
   ___GLOBALS_MULTIPROCESS_TEST.holes = holes
   ___GLOBALS_MULTIPROCESS_TEST.plugs = plugs
   make_sampler = plug_get_sample_hierarchy
   if args.trial_run:
      make_sampler = plug_test_hier_sampler
   t = perf_counter()
   with ProcessPoolExecutor(args.nprocess) as exe:
      futures = [
         exe.submit(multiprocess_helper, ih, ip, make_sampler, args)
         for ih in range(len(holes))
         for ip in range(len(plugs))
      ]
      results = [f.result() for f in futures]
   print(
      "search time",
      perf_counter() - t,
      "nprocess",
      args.nprocess,
      "nthread",
      args.nthread,
   )
   return concat_results(results)

def server_test(cli_args):
   args = rpxdock.app.defaults()
   args.nout_debug = 3
   args.nresl = 5
   args.wts = Bunch(plug=1.0, hole=1.0, ncontact=0.1, rpx=1.0)
   args.beam_size = 1e4
   args.max_bb_redundancy = 3.0
   args.max_longaxis_dot_z = 0.5
   args.executor = ThreadPoolExecutor(args.nworker)
   args.multi_iface_summary = np.min  # min(plug, hole)
   # args.max_longaxis_dot_z = 0.1
   # args.multi_iface_summary = np.sum  # sum(plug, hole)
   # args.multi_iface_summary = lambda x, **kw: x[:, 0]  # plug only
   # args.wts = Bunch(plug=1.0, hole=1.0, ncontact=1.0, rpx=0)  # ncontact only
   # args.output_prefix = "rpx"

   make_sampler = plug_get_sample_hierarchy
   args = args.sub_(nout_debug=20, beam_size=3e5, max_bb_redundancy=3)
   # hscore_tables = HC(load_big_hscore)
   hscore_tables = HC(load_medium_hscore)
   # args = args.sub_(nout_debug=2, beam_size=2e4, max_bb_redundancy=3)
   # hscore_tables = HC(load_small_hscore)
   hscore = RpxHier(hscore_tables)
   holes = [
      ("C3_o42", HC(Body, "/home/sheffler/scaffolds/holes/C3_o42_Z_asym.pdb", sym=3)),
      ("C3_i52", HC(Body, "/home/sheffler/scaffolds/holes/C3_i52_Z_asym.pdb", sym=3)),
   ]
   # plug = HC(Body, datadir + "/pdb/DHR14.pdb", n=0)
   plg = ["DHR01", "DHR03", "DHR04", "DHR05", "DHR07", "DHR08", "DHR09", "DHR10"]
   plg += ["DHR14", "DHR15", "DHR18", "DHR20", "DHR21", "DHR23", "DHR24"]
   plg += ["DHR26", "DHR27", "DHR31", "DHR32", "DHR36", "DHR39", "DHR46"]
   plg += ["DHR47", "DHR49", "DHR52", "DHR53", "DHR54", "DHR55", "DHR57"]
   plg += ["DHR58", "DHR59", "DHR62", "DHR64", "DHR68", "DHR70", "DHR71"]
   plg += ["DHR72", "DHR76", "DHR77", "DHR78", "DHR79", "DHR80", "DHR81"]
   plg += ["DHR82"]
   d = "/home/sheffler/scaffolds/repeat/dhr8/"
   for htag, hole in holes:
      dump_pdb_from_bodies("hole_%s.pdb" % htag, [hole], symframes(hole.sym))
      for ptag in plg:
         fname = d + ptag + ".pdb"
         plug = HC(Body, fname, n=0)
         pre = htag + "_" + ptag
         sampler = make_sampler(plug, hole, hscore)
         make_plugs(plug, hole, hscore, sampler, output_prefix=pre, **kws)

def hier_sample_test(cli_args):
   args = rpxdock.app.defaults()
   args.nout_debug = 3
   args.nresl = 5
   args.wts = Bunch(plug=1.0, hole=1.0, ncontact=0.1, rpx=1.0)
   args.beam_size = 1e4
   args.max_bb_redundancy = 3.0
   args.max_longaxis_dot_z = 0.5
   args.executor = ThreadPoolExecutor(args.nworker)
   args.multi_iface_summary = np.min  # min(plug, hole)
   # args.max_longaxis_dot_z = 0.1
   # args.multi_iface_summary = np.sum  # sum(plug, hole)
   # args.multi_iface_summary = lambda x, **kw: x[:, 0]  # plug only
   # args.wts = Bunch(plug=1.0, hole=1.0, ncontact=1.0, rpx=0)  # ncontact only
   # args.output_prefix = "rpx"

   raise NotImplemented
   hscore = RpxHier(load_big_hscore())
   args.wts.ncontact = 1.0
   args.wts.rpx = 1.0
   args.beam_size = 2e5
   args.nout_debug = 20
   args.multi_iface_summary = np.sum
   args.output_prefix = "ncontact_over_rpx_sum"
   smapler = plug_get_sample_hierarchy(plug, hole, hscore)
   # make_plugs(plug, hole, hscore, **kws)
   __make_plugs_hier_sample_test__(plug, hole, hscore, **kws)

# @click.command()
# @click.argument("MODE", default="quick_test")
# @click.option("--nprocess", default=1)
# @click.option("--nthread", default=1)
# def main(MODE, **kw):
#     if MODE == "quick_test":
#         quick_test(**kw)
#
#     elif MODE == "multiprocess_test":
#         multiprocess_test(**kw)
#
#     elif MODE == "test_server":
#         server_test(**kw)
#
#     elif MODE == "hier_sample_test":
#         hier_sample_test(**kw)

def load_small_hscore():
   return load_threads(small_hscore_fnames)

def load_medium_hscore():
   return load_threads(medium_hscore_fnames)

def load_big_hscore():
   return load_threads(big_hscore_fnames)

def main():
   rpxdock.rosetta.rosetta_init()
   if len(sys.argv) is 1:
      hscore = RpxHier(load_threads(small_hscore_fnames))
      return quick_test_olig(hscore)
   parser = rpxdock.app.default_cli_parser()
   parser.add_argument("mode", default="quick_test")
   parser.add_argument("--plugs", nargs="+")
   parser.add_argument("--holes", nargs="+")
   cli_args = parser.parse_args()
   mode = cli_args.mode
   if mode in globals():
      return globals()[mode](cli_args)
   print("unknown mode " + mode)

if __name__ == "__main__":
   main()
