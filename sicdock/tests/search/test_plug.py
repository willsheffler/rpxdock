from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
import _pickle, threading, os, argparse, sys
from time import perf_counter
import numpy as np
import click
import sicdock
from sicdock.motif import HierScore
from sicdock.motif._loadhack import hackcache as HC
from sicdock.data import datadir
from sicdock.util import load, Bunch
from sicdock.body import Body
from sicdock.io.io_body import dump_pdb_from_bodies
from sicdock.sym import symframes
from sicdock.search.plug import (
    make_plugs,
    plug_get_sample_hierarchy,
    __make_plugs_hier_sample_test__,
    ____PLUG_TEST_SAMPLE_HIERARCHY____,
)

__FORK_GLOBALS__ = Bunch()


def multiprocess_helper(ihole, iplug, make_sampler, args):
    global __FORK_GLOBALS__
    hscore = __FORK_GLOBALS__.hscore
    htag, hole = __FORK_GLOBALS__.holes[ihole]
    ptag, plug = __FORK_GLOBALS__.plugs[iplug]
    tag = f"{htag}_{ptag}"
    sampler = make_sampler(plug, hole, hscore)
    if args.nthread > 1:
        args.executor = ThreadPoolExecutor(args.nthread)
    make_plugs(plug, hole, hscore, sampler, out_prefix=tag, **args)
    if args.executor:
        args.executor.shutdown(wait=False)


def load_file(f):
    tag = os.path.basename(f).replace(".pdb", "")
    if tag[0] == "C" and tag[2] == "_":
        body = Body(f, sym=tag[:2])
    else:
        body = Body(f)
    return tag, body


def multiprocess_test(cli_args):

    args = sicdock.options.defaults()
    args.nout = 1
    args.nresl = 5
    args.wts = Bunch(plug=1.0, hole=1.0, ncontact=0.1, rpx=1.0)
    args.beam_size = 1e4
    args.rmscut = 3.0
    args.max_longaxis_dot_z = 0.5
    args.multi_iface_summary = np.min  # min(plug, hole)
    args.reduced_sampling = True
    d = "/home/sheffler/scaffolds/repeat/dhr8/"
    p = ["DHR01", "DHR03", "DHR04", "DHR05", "DHR07", "DHR08", "DHR09", "DHR10"]
    args.plugs = [d + x + ".pdb" for x in p]
    args.holes = [
        "/home/sheffler/scaffolds/holes/C3_i52_Z_asym.pdb",
        "/home/sheffler/scaffolds/holes/C3_o42_Z_asym.pdb",
    ]
    args = args.sub(vars(cli_args))

    if len(args.hscore_files) is 1 and args.hscore_files[0] in globals():
        args.hscore_files = globals()[args.hscore_files[0]]
    loader = MultiLoader(args.hscore_files)
    loader.start()
    with ProcessPoolExecutor(max(args.nthread, args.nprocess)) as exe:
        bodies = list(exe.map(load_file, args.holes + args.plugs))
    holes = bodies[: len(args.holes)]
    for htag, hole in holes:
        htag = args.out_prefix + htag if args.out_prefix else htag
        dump_pdb_from_bodies(htag + ".pdb", [hole], symframes(hole.sym))
    plugs = bodies[len(args.holes) :]
    loader.join()
    hscore = HierScore(loader.result)

    global __FORK_GLOBALS__
    __FORK_GLOBALS__.hscore = hscore
    __FORK_GLOBALS__.holes = holes
    __FORK_GLOBALS__.plugs = plugs
    make_sampler = plug_get_sample_hierarchy
    if args.reduced_sampling:
        make_sampler = ____PLUG_TEST_SAMPLE_HIERARCHY____
    t = perf_counter()
    with ProcessPoolExecutor(args.nprocess) as exe:
        futures = [
            exe.submit(multiprocess_helper, ih, ip, make_sampler, args)
            for ih in range(len(holes))
            for ip in range(len(plugs))
        ]
        [f.result() for f in futures]
    print(
        "search time",
        perf_counter() - t,
        "nprocess",
        args.nprocess,
        "nthread",
        args.nthread,
    )


def quick_test(cli_args):
    args = sicdock.options.defaults()
    args.nout = 3
    args.nresl = 5
    args.wts = Bunch(plug=1.0, hole=1.0, ncontact=0.1, rpx=1.0)
    args.beam_size = 1e4
    args.rmscut = 3.0
    args.max_longaxis_dot_z = 0.5
    args.executor = ThreadPoolExecutor(args.nthread)
    args.multi_iface_summary = np.min  # min(plug, hole)
    # args.max_longaxis_dot_z = 0.1
    # args.multi_iface_summary = np.sum  # sum(plug, hole)
    # args.multi_iface_summary = lambda x, **kw: x[:, 0]  # plug only
    # args.wts = Bunch(plug=1.0, hole=1.0, ncontact=1.0, rpx=0)  # ncontact only
    # args.out_prefix = "rpx"
    args.sub(vars(cli_args))

    make_sampler = ____PLUG_TEST_SAMPLE_HIERARCHY____
    try:
        with open("/home/sheffler/debug/sicdock/hole.pickle", "rb") as inp:
            plug, hole = _pickle.load(inp)
        # plug = sicdock.body.Body("/home/sheffler/scaffolds/repeat/dhr8/DHR05.pdb")
    except:
        plug = sicdock.body.Body(datadir + "/pdb/DHR14.pdb")
        # hole = sicdock.body.Body(datadir + "/pdb/hole_C3_tiny.pdb", sym=3)
        hole = HC(Body, "/home/sheffler/scaffolds/holes/C3_i52_Z_asym.pdb", sym=3)
        with open("/home/sheffler/debug/sicdock/hole.pickle", "wb") as out:
            _pickle.dump([plug, hole], out)
    # nresl = 2
    hscore = HierScore(load_small_hscore())
    sampler = make_sampler(plug, hole, hscore)
    make_plugs(plug, hole, hscore, sampler, **args)


def server_test(cli_args):
    args = sicdock.options.defaults()
    args.nout = 3
    args.nresl = 5
    args.wts = Bunch(plug=1.0, hole=1.0, ncontact=0.1, rpx=1.0)
    args.beam_size = 1e4
    args.rmscut = 3.0
    args.max_longaxis_dot_z = 0.5
    args.executor = ThreadPoolExecutor(args.nworker)
    args.multi_iface_summary = np.min  # min(plug, hole)
    # args.max_longaxis_dot_z = 0.1
    # args.multi_iface_summary = np.sum  # sum(plug, hole)
    # args.multi_iface_summary = lambda x, **kw: x[:, 0]  # plug only
    # args.wts = Bunch(plug=1.0, hole=1.0, ncontact=1.0, rpx=0)  # ncontact only
    # args.out_prefix = "rpx"

    make_sampler = plug_get_sample_hierarchy
    args = args.sub_(nout=20, beam_size=3e5, rmscut=3)
    # hscore_tables = HC(load_big_hscore)
    hscore_tables = HC(load_medium_hscore)
    # args = args.sub_(nout=2, beam_size=2e4, rmscut=3)
    # hscore_tables = HC(load_small_hscore)
    hscore = HierScore(hscore_tables)
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
            make_plugs(plug, hole, hscore, sampler, out_prefix=pre, **args)


def hier_sample_test(cli_args):
    args = sicdock.options.defaults()
    args.nout = 3
    args.nresl = 5
    args.wts = Bunch(plug=1.0, hole=1.0, ncontact=0.1, rpx=1.0)
    args.beam_size = 1e4
    args.rmscut = 3.0
    args.max_longaxis_dot_z = 0.5
    args.executor = ThreadPoolExecutor(args.nworker)
    args.multi_iface_summary = np.min  # min(plug, hole)
    # args.max_longaxis_dot_z = 0.1
    # args.multi_iface_summary = np.sum  # sum(plug, hole)
    # args.multi_iface_summary = lambda x, **kw: x[:, 0]  # plug only
    # args.wts = Bunch(plug=1.0, hole=1.0, ncontact=1.0, rpx=0)  # ncontact only
    # args.out_prefix = "rpx"

    raise NotImplemented
    hscore = HierScore(load_big_hscore())
    args.wts.ncontact = 1.0
    args.wts.rpx = 1.0
    args.beam_size = 2e5
    args.nout = 20
    args.multi_iface_summary = np.sum
    args.out_prefix = "ncontact_over_rpx_sum"
    smapler = plug_get_sample_hierarchy(plug, hole, hscore)
    # make_plugs(plug, hole, hscore, **args)
    __make_plugs_hier_sample_test__(plug, hole, hscore, **args)


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


class Loader(threading.Thread):
    def __init__(self, fname):
        super().__init__(None, None, None)
        self.fname = fname

    def run(self):
        self.result = load(self.fname)
        print("loaded", self.fname)


class MultiLoader(threading.Thread):
    def __init__(self, fnames):
        super().__init__(None, None, None)
        self.fnames = fnames

    def run(self):
        threads = [Loader(f) for f in self.fnames]
        [t.start() for t in threads]
        [t.join() for t in threads]
        self.result = [t.result for t in threads]


big_hscore_fnames = [
    "/home/sheffler/debug/sicdock/respairdat/hscore/pdb_res_pair_data_si30_rots_SS_p0.5_b1_base.pickle",
    # "/home/sheffler/debug/sicdock/respairdat/hscore/pdb_res_pair_data_si30_rots_SS_p0.5_b1_hier0_Kflat_2_1.pickle", #  2.2G
    "/home/sheffler/debug/sicdock/respairdat/hscore/pdb_res_pair_data_si30_rots_SS_p0.5_b1_hier0_Kflat_3_0.pickle",  #  3.7G
    # "/home/sheffler/debug/sicdock/respairdat/hscore/pdb_res_pair_data_si30_rots_SS_p0.5_b1_hier1_Kflat_1_1.pickle", # 1012M
    "/home/sheffler/debug/sicdock/respairdat/hscore/pdb_res_pair_data_si30_rots_SS_p0.5_b1_hier1_Kflat_2_0.pickle",  #  3.1G
    # "/home/sheffler/debug/sicdock/respairdat/hscore/pdb_res_pair_data_si30_rots_SS_p0.5_b1_hier1_Kflat_2_1.pickle", #  6.1G
    "/home/sheffler/debug/sicdock/respairdat/hscore/pdb_res_pair_data_si30_rots_SS_p0.5_b1_hier2_Kflat_1_0.pickle",  #  2.3G
    # "/home/sheffler/debug/sicdock/respairdat/hscore/pdb_res_pair_data_si30_rots_SS_p0.5_b1_hier2_Kflat_1_1.pickle", #  7.6G
    "/home/sheffler/debug/sicdock/respairdat/hscore/pdb_res_pair_data_si30_rots_SS_p0.5_b1_hier3_Kflat_1_0.pickle",  #  4.2G
    "/home/sheffler/debug/sicdock/respairdat/hscore/pdb_res_pair_data_si30_rots_SS_p0.5_b1_hier4_Kflat_1_0.pickle",  #  6.2G
]


def load_big_hscore():
    print("========================= LOADING BIG H-SCORES ===========================")
    #  1.1G May  5 pdb_res_pair_data_si30_rots_SS_p0.5_b1_base.pickle
    # hier0  2.2G May  4 pdb_res_pair_data_si30_rots_SS_p0.5_b1_hier0_Kflat_2_1.pickle
    # hier0  3.7G May  5 pdb_res_pair_data_si30_rots_SS_p0.5_b1_hier0_Kflat_3_0.pickle
    # hier1 1012M May  4 pdb_res_pair_data_si30_rots_SS_p0.5_b1_hier1_Kflat_1_1.pickle
    # hier1  3.1G May  5 pdb_res_pair_data_si30_rots_SS_p0.5_b1_hier1_Kflat_2_0.pickle
    # hier1  6.1G May  5 pdb_res_pair_data_si30_rots_SS_p0.5_b1_hier1_Kflat_2_1.pickle
    # hier2  2.3G May  4 pdb_res_pair_data_si30_rots_SS_p0.5_b1_hier2_Kflat_1_0.pickle
    # hier2  7.6G May  5 pdb_res_pair_data_si30_rots_SS_p0.5_b1_hier2_Kflat_1_1.pickle
    # hier3  4.2G May  5 pdb_res_pair_data_si30_rots_SS_p0.5_b1_hier3_Kflat_1_0.pickle
    # hier4  6.2G May  5 pdb_res_pair_data_si30_rots_SS_p0.5_b1_hier4_Kflat_1_0.pickle
    threads = [Loader(path + f) for f in big_hscore_fnames]
    [t.start() for t in threads]
    [t.join() for t in threads]
    return [t.result for t in threads]


medium_hscore_fnames = [
    "/home/sheffler/debug/sicdock/respairdat/hscore/pdb_res_pair_data_si30_1000_rots_SS_p0.5_b1_base.pickle",  #  68M
    "/home/sheffler/debug/sicdock/respairdat/hscore/pdb_res_pair_data_si30_1000_rots_SS_p0.5_b1_hier0_Kflat_3_0.pickle",  # 1.8G
    "/home/sheffler/debug/sicdock/respairdat/hscore/pdb_res_pair_data_si30_1000_rots_SS_p0.5_b1_hier1_Kflat_2_0.pickle",  # 1.2G
    "/home/sheffler/debug/sicdock/respairdat/hscore/pdb_res_pair_data_si30_1000_rots_SS_p0.5_b1_hier2_Kflat_1_1.pickle",  # 1.7G
    "/home/sheffler/debug/sicdock/respairdat/hscore/pdb_res_pair_data_si30_1000_rots_SS_p0.5_b1_hier3_Kflat_1_0.pickle",  # 688M
    "/home/sheffler/debug/sicdock/respairdat/hscore/pdb_res_pair_data_si30_1000_rots_SS_p0.5_b1_hier4_Kflat_1_0.pickle",  # 819M
]


def load_medium_hscore():
    print("======================= LOADING MEDIUM H-SCORES =========================")
    threads = [Loader(path + f) for f in medium_hscore_fnames]
    [t.start() for t in threads]
    [t.join() for t in threads]
    return [t.result for t in threads]


small_noss_hscore_fnames = [
    "/home/sheffler/debug/sicdock/respairdat/hscore/pdb_res_pair_data_si30_10_rots_noSS_p0.5_b1_base.pickle",
    "/home/sheffler/debug/sicdock/respairdat/hscore/pdb_res_pair_data_si30_10_rots_noSS_p0.5_b1_hier0_Kflat_1_1.pickle",
    "/home/sheffler/debug/sicdock/respairdat/hscore/pdb_res_pair_data_si30_10_rots_noSS_p0.5_b1_hier1_Kflat_1_1.pickle",
    "/home/sheffler/debug/sicdock/respairdat/hscore/pdb_res_pair_data_si30_10_rots_noSS_p0.5_b1_hier2_Kflat_1_1.pickle",
    "/home/sheffler/debug/sicdock/respairdat/hscore/pdb_res_pair_data_si30_10_rots_noSS_p0.5_b1_hier3_Kflat_1_1.pickle",
    "/home/sheffler/debug/sicdock/respairdat/hscore/pdb_res_pair_data_si30_10_rots_noSS_p0.5_b1_hier4_Kflat_1_1.pickle",
]


def load_small_noss_hscore():
    print("======================= LOADING SMALL H-SCORES =========================")
    threads = [Loader(path + f) for f in small_noss_hscore_fnames]
    [t.start() for t in threads]
    [t.join() for t in threads]
    return [t.result for t in threads]


small_hscore_fnames = [
    "/home/sheffler/debug/sicdock/hscore_small_ss/pdb_res_pair_data_si30_10_rots_SS_p0.5_b1_base.pickle",
    "/home/sheffler/debug/sicdock/hscore_small_ss/pdb_res_pair_data_si30_10_rots_SS_p0.5_b1_hier0_Kflat_1_0.pickle",
    "/home/sheffler/debug/sicdock/hscore_small_ss/pdb_res_pair_data_si30_10_rots_SS_p0.5_b1_hier1_Kflat_1_0.pickle",
    "/home/sheffler/debug/sicdock/hscore_small_ss/pdb_res_pair_data_si30_10_rots_SS_p0.5_b1_hier2_Kflat_1_0.pickle",
    "/home/sheffler/debug/sicdock/hscore_small_ss/pdb_res_pair_data_si30_10_rots_SS_p0.5_b1_hier3_Kflat_1_0.pickle",
    "/home/sheffler/debug/sicdock/hscore_small_ss/pdb_res_pair_data_si30_10_rots_SS_p0.5_b1_hier4_Kflat_1_0.pickle",
]


def load_small_hscore():
    print("======================= LOADING SMALL SS H-SCORES =========================")
    threads = [Loader(path + f) for f in small_hscore_fnames]
    [t.start() for t in threads]
    [t.join() for t in threads]
    return [t.result for t in threads]


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("mode", default="quick_test")
    parser.add_argument("--nthread", type=int)
    parser.add_argument("--nprocess", type=int)
    parser.add_argument("--beam_size", type=int, default=10000)
    parser.add_argument("--reduced_sampling", default=True)
    parser.add_argument("--plugs", nargs="+")
    parser.add_argument("--holes", nargs="+")
    parser.add_argument("--hscore_files", nargs="+")
    cli_args = parser.parse_args()
    mode = cli_args.mode

    if len(sys.argv) is 1:
        quick_test()
    elif mode in vars():
        vars()[mode](cli_args)
    else:
        print("unknown mode " + mode)
