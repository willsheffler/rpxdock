from concurrent.futures import ThreadPoolExecutor
import _pickle, threading
import sicdock
from sicdock.motif import HierScore
from sicdock.motif._loadhack import hackcache as HC
from sicdock.search.plug import make_plugs
from sicdock.data import datadir
from sicdock.util import load, Bunch
from sicdock.body import Body

# iresl 4 ntot      99,968 nonzero 80,476
# redundancy filter cut 3 keep 211 of None
# DHR14_00.pdb score  69.743 olig:  69.651 hole:  75.253 resi 0-306
# DHR14_01.pdb score  67.688 olig:  67.595 hole:  71.522 resi 0-256
# DHR14_02.pdb score  67.459 olig:  67.384 hole:  75.708 resi 0-259
# DHR14_03.pdb score  66.320 olig:  66.252 hole:  68.275 resi 0-265
# DHR14_04.pdb score  65.580 olig:  72.296 hole:  65.431 resi 0-296
# DHR14_05.pdb score  65.525 olig:  65.472 hole:  67.479 resi 0-307
# DHR14_06.pdb score  65.296 olig:  65.190 hole:  71.125 resi 0-297
# DHR14_07.pdb score  63.663 olig:  64.107 hole:  63.568 resi 0-274
# DHR14_08.pdb score  63.553 olig:  63.446 hole:  75.987 resi 0-293
# DHR14_09.pdb score  62.911 olig:  67.863 hole:  62.827 resi 0-302
# DHR14_10.pdb score  62.714 olig:  62.625 hole:  78.431 resi 0-279
# DHR14_11.pdb score  62.248 olig:  62.209 hole:  62.118 resi 0-297
# DHR14_12.pdb score  61.747 olig:  61.691 hole:  61.987 resi 0-318
# DHR14_13.pdb score  60.718 olig:  65.501 hole:  60.622 resi 0-306
# DHR14_14.pdb score  60.323 olig:  61.117 hole:  60.218 resi 0-265
# DHR14_15.pdb score  59.946 olig:  59.871 hole:  60.879 resi 0-302
# DHR14_16.pdb score  59.768 olig:  59.702 hole:  70.176 resi 0-257
# DHR14_17.pdb score  59.080 olig:  61.783 hole:  58.966 resi 0-303
# DHR14_18.pdb score  58.366 olig:  58.287 hole:  60.754 resi 0-299
# DHR14_19.pdb score  58.206 olig:  65.049 hole:  58.092 resi 0-280
# samprate: 23,842/s perf ttot 164.743 tdump   1.015
# eval time per stage:   125.65s     7.67s     8.62s     9.54s    11.55s
# eval rate per stage:    28,078/s  13,041/s  11,592/s  10,481/s   8,654/s
# exe time 281.9740585499967
#

# XformHier 6,498,000 [19 19  6] 30.64299964904785 [-114. -114.  -36.] [114. 114.  36.]
# ntasks 649
# iresl 0 ntot   6,498,000 nonzero 15,665
# ntasks 9
# iresl 1 ntot      99,968 nonzero 26,137
# ntasks 9
# iresl 2 ntot      99,968 nonzero 43,119
# ntasks 9
# iresl 3 ntot      99,872 nonzero 60,325
# ntasks 9
# iresl 4 ntot      99,904 nonzero 74,492
# redundancy filter cut 3 keep 217 of None
# DHR14_00.pdb score  54.689 olig:  64.831 hole:  54.632 resi 0-319
# DHR14_01.pdb score  52.090 olig:  53.996 hole:  52.033 resi 0-319
# DHR14_02.pdb score  50.397 olig:  50.674 hole:  50.315 resi 0-319
# DHR14_03.pdb score  48.555 olig:  55.653 hole:  48.517 resi 0-319
# DHR14_04.pdb score  47.690 olig:  47.585 hole:  48.013 resi 0-319
# DHR14_05.pdb score  43.291 olig:  43.208 hole:  43.831 resi 0-319
# DHR14_06.pdb score  43.033 olig:  44.657 hole:  42.902 resi 0-319
# DHR14_07.pdb score  42.994 olig:  45.577 hole:  42.954 resi 0-319
# DHR14_08.pdb score  41.571 olig:  44.498 hole:  41.530 resi 0-319
# DHR14_09.pdb score  40.556 olig:  41.677 hole:  40.518 resi 0-319
# DHR14_10.pdb score  40.377 olig:  41.633 hole:  40.342 resi 0-319
# DHR14_11.pdb score  39.475 olig:  43.957 hole:  39.391 resi 0-319
# DHR14_12.pdb score  39.132 olig:  39.334 hole:  39.064 resi 0-319
# DHR14_13.pdb score  38.946 olig:  45.303 hole:  38.870 resi 0-319
# DHR14_14.pdb score  38.691 olig:  38.619 hole:  39.453 resi 0-319
# DHR14_15.pdb score  38.297 olig:  38.241 hole:  41.292 resi 0-319
# DHR14_16.pdb score  38.177 olig:  38.080 hole:  41.355 resi 0-319
# DHR14_17.pdb score  38.176 olig:  42.665 hole:  38.132 resi 0-319
# DHR14_18.pdb score  38.088 olig:  42.511 hole:  38.050 resi 0-319
# DHR14_19.pdb score  37.744 olig:  38.483 hole:  37.710 resi 0-319
# samprate: 101,284/s perf ttot  68.103 tdump   1.105
# eval time per stage:    44.88s     3.16s     4.59s     5.84s     7.23s
# eval rate per stage:   144,784/s  31,586/s  21,796/s  17,088/s  13,816/s
# exe time 68.10635681598797


def main():
    o = sicdock.Bunch(
        nworker=8,
        max_trim=100,
        nout=3,
        nresl=5,
        wts=Bunch(plug=1.0, hole=1.0, ncontact=0.001, rpx=1.0),
        clashdis=3.5,
        beam_size=1e4,
        rmscut=3.0,
        max_cluster=None,
        TEST=True,
    )
    exe = ThreadPoolExecutor(o.nworker)

    quick = False
    if quick:
        try:
            with open("/home/sheffler/debug/sicdock/hole.pickle", "rb") as inp:
                plug, hole = _pickle.load(inp)
        except:
            plug = sicdock.body.Body(datadir + "/pdb/DHR14.pdb")
            # hole = sicdock.body.Body(datadir + "/pdb/hole_C3_tiny.pdb", sym=3)
            hole = HC(Body, "/home/sheffler/scaffolds/holes/C3_i52_Z_asym.pdb", sym=3)
            with open("/home/sheffler/debug/sicdock/hole.pickle", "wb") as out:
                _pickle.dump([plug, hole], out)
        hscore = HierScore(load_small_hscore())
        make_plugs(plug, hole, hscore, exe=exe, **o)
    else:
        o = o.with_(nout=20, beam_size=3e5, TEST=False, rmscut=3)
        # hscore_tables = HC(load_big_hscore)
        # hscore_tables = HC(load_medium_hscore)

        o = o.with_(nout=2, beam_size=2e4, TEST=True, rmscut=3)
        hscore_tables = HC(load_small_hscore)

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
            for ptag in plg:
                fname = d + ptag + ".pdb"
                plug = HC(Body, fname, n=0)
                pre = htag + "_" + ptag
                make_plugs(plug, hole, hscore, exe=exe, out_prefix=pre, **o)

    exe.shutdown(wait=False)


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
