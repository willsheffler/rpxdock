from concurrent.futures import ThreadPoolExecutor
import _pickle, threading
import numpy as np
import sicdock
from sicdock.motif import HierScore
from sicdock.motif._loadhack import hackcache as HC
from sicdock.search.plug import make_plugs, __make_plugs_hier_sample_test__
from sicdock.data import datadir
from sicdock.util import load, Bunch
from sicdock.body import Body
from sicdock.io.io_body import dump_pdb_from_bodies
from sicdock.sym import symframes


def main():
    args = sicdock.options.defaults()
    args.nout = 3
    args.nresl = 5
    args.wts = Bunch(plug=1.0, hole=1.0, ncontact=0.1, rpx=1.0)
    args.beam_size = 1e4
    args.rmscut = 3.0
    args.max_longaxis_dot_z = 0.5
    args.TEST = True
    args.exe = ThreadPoolExecutor(args.nworker)
    args.multi_iface_summary = np.min  # min(plug, hole)

    # args.max_longaxis_dot_z = 0.1
    # args.multi_iface_summary = np.sum  # sum(plug, hole)
    # args.multi_iface_summary = lambda x, **kw: x[:, 0]  # plug only
    # args.wts = Bunch(plug=1.0, hole=1.0, ncontact=1.0, rpx=0)  # ncontact only
    # args.out_prefix = "rpx"

    if args.TEST:
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
        make_plugs(plug, hole, hscore, **args.sub(TEST=0))

        # hscore = HierScore(load_big_hscore())
        # args.wts.ncontact = 1.0
        # args.wts.rpx = 1.0
        # args.beam_size = 2e5
        # args.nout = 20
        # args.multi_iface_summary = np.sum
        # args.out_prefix = "ncontact_over_rpx_sum"
        # args.TEST = False
        # make_plugs(plug, hole, hscore, **args)
        # # __make_plugs_hier_sample_test__(plug, hole, hscore, **args)
    else:
        args = args.sub_(nout=20, beam_size=3e5, rmscut=3)
        # hscore_tables = HC(load_big_hscore)
        hscore_tables = HC(load_medium_hscore)

        # args = args.sub_(nout=2, beam_size=2e4, TEST=True, rmscut=3)
        # hscore_tables = HC(load_small_hscore)

        hscore = HierScore(hscore_tables)
        holes = [
            (
                "C3_o42",
                HC(Body, "/home/sheffler/scaffolds/holes/C3_o42_Z_asym.pdb", sym=3),
            ),
            (
                "C3_i52",
                HC(Body, "/home/sheffler/scaffolds/holes/C3_i52_Z_asym.pdb", sym=3),
            ),
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
                make_plugs(plug, hole, hscore, out_prefix=pre, **args)

    args.exe.shutdown(wait=False)


class Loader(threading.Thread):
    def __init__(self, fname):
        super().__init__(None, None, None)
        self.fname = fname

    def run(self):
        self.result = load(self.fname)
        print("loaded", self.fname)


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

    fnames = [
        "pdb_res_pair_data_si30_rots_SS_p0.5_b1_base.pickle",
        # "pdb_res_pair_data_si30_rots_SS_p0.5_b1_hier0_Kflat_2_1.pickle", #  2.2G
        "pdb_res_pair_data_si30_rots_SS_p0.5_b1_hier0_Kflat_3_0.pickle",  #  3.7G
        # "pdb_res_pair_data_si30_rots_SS_p0.5_b1_hier1_Kflat_1_1.pickle", # 1012M
        "pdb_res_pair_data_si30_rots_SS_p0.5_b1_hier1_Kflat_2_0.pickle",  #  3.1G
        # "pdb_res_pair_data_si30_rots_SS_p0.5_b1_hier1_Kflat_2_1.pickle", #  6.1G
        "pdb_res_pair_data_si30_rots_SS_p0.5_b1_hier2_Kflat_1_0.pickle",  #  2.3G
        # "pdb_res_pair_data_si30_rots_SS_p0.5_b1_hier2_Kflat_1_1.pickle", #  7.6G
        "pdb_res_pair_data_si30_rots_SS_p0.5_b1_hier3_Kflat_1_0.pickle",  #  4.2G
        "pdb_res_pair_data_si30_rots_SS_p0.5_b1_hier4_Kflat_1_0.pickle",  #  6.2G
    ]
    path = "/home/sheffler/debug/sicdock/respairdat/hscore/"
    threads = [Loader(path + f) for f in fnames]
    [t.start() for t in threads]
    [t.join() for t in threads]
    return [t.result for t in threads]


def load_medium_hscore():
    print("======================= LOADING MEDIUM H-SCORES =========================")

    fnames = [
        "pdb_res_pair_data_si30_1000_rots_SS_p0.5_b1_base.pickle",  #  68M
        "pdb_res_pair_data_si30_1000_rots_SS_p0.5_b1_hier0_Kflat_3_0.pickle",  # 1.8G
        "pdb_res_pair_data_si30_1000_rots_SS_p0.5_b1_hier1_Kflat_2_0.pickle",  # 1.2G
        "pdb_res_pair_data_si30_1000_rots_SS_p0.5_b1_hier2_Kflat_1_1.pickle",  # 1.7G
        "pdb_res_pair_data_si30_1000_rots_SS_p0.5_b1_hier3_Kflat_1_0.pickle",  # 688M
        "pdb_res_pair_data_si30_1000_rots_SS_p0.5_b1_hier4_Kflat_1_0.pickle",  # 819M
    ]
    path = "/home/sheffler/debug/sicdock/respairdat/hscore/"
    threads = [Loader(path + f) for f in fnames]
    [t.start() for t in threads]
    [t.join() for t in threads]
    return [t.result for t in threads]


def load_small_noss_hscore():
    print("======================= LOADING SMALL H-SCORES =========================")

    fnames = [
        "pdb_res_pair_data_si30_10_rots_noSS_p0.5_b1_base.pickle",
        "pdb_res_pair_data_si30_10_rots_noSS_p0.5_b1_hier0_Kflat_1_1.pickle",
        "pdb_res_pair_data_si30_10_rots_noSS_p0.5_b1_hier1_Kflat_1_1.pickle",
        "pdb_res_pair_data_si30_10_rots_noSS_p0.5_b1_hier2_Kflat_1_1.pickle",
        "pdb_res_pair_data_si30_10_rots_noSS_p0.5_b1_hier3_Kflat_1_1.pickle",
        "pdb_res_pair_data_si30_10_rots_noSS_p0.5_b1_hier4_Kflat_1_1.pickle",
    ]
    path = "/home/sheffler/debug/sicdock/respairdat/hscore/"
    threads = [Loader(path + f) for f in fnames]
    [t.start() for t in threads]
    [t.join() for t in threads]
    return [t.result for t in threads]


def load_small_hscore():
    print("======================= LOADING SMALL SS H-SCORES =========================")

    fnames = [
        "pdb_res_pair_data_si30_10_rots_SS_p0.5_b1_base.pickle",
        "pdb_res_pair_data_si30_10_rots_SS_p0.5_b1_hier0_Kflat_1_0.pickle",
        "pdb_res_pair_data_si30_10_rots_SS_p0.5_b1_hier1_Kflat_1_0.pickle",
        "pdb_res_pair_data_si30_10_rots_SS_p0.5_b1_hier2_Kflat_1_0.pickle",
        "pdb_res_pair_data_si30_10_rots_SS_p0.5_b1_hier3_Kflat_1_0.pickle",
        "pdb_res_pair_data_si30_10_rots_SS_p0.5_b1_hier4_Kflat_1_0.pickle",
    ]
    path = "/home/sheffler/debug/sicdock/hscore_small_ss/"
    threads = [Loader(path + f) for f in fnames]
    [t.start() for t in threads]
    [t.join() for t in threads]
    return [t.result for t in threads]


if __name__ == "__main__":
    main()


# -O1
# plug_00.pdb score  17.137 olig:   8.807 hole:   5.937 resi 20-319     102     112
# plug_01.pdb score  17.057 olig:   9.215 hole:  10.857 resi 0-319      85      62
# plug_02.pdb score  17.019 olig:  10.275 hole:   5.119 resi 0-319      88     119
# rate: 36,699/s ttot  97.222 tdump   0.347
# stage time:    93.19s     0.55s     0.69s     0.79s     1.03s
# stage rate:    37,856/s  18,095/s  14,388/s  12,596/s   9,688/s
# ======================== runtests.py done, time  99.970 ========================

# -Ofast
# plug_00.pdb score  17.137 olig:   8.807 hole:   5.937 resi 20-319     102     112
# plug_01.pdb score  17.057 olig:   9.215 hole:  10.857 resi 0-319      85      62
# plug_02.pdb score  17.019 olig:  10.275 hole:   5.119 resi 0-319      88     119
# rate: 47,136/s ttot  75.694 tdump   0.286
# stage time:    72.46s     0.44s     0.53s     0.63s     0.70s
# stage rate:    48,687/s  22,729/s  18,723/s  15,759/s  14,346/s
# ======================== runtests.py done, time  79.447 ========================
