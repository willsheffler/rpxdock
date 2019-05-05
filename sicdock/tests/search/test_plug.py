import sicdock
from sicdock.motif import HierScore
from sicdock.motif._loadhack import hackcache as HC
from sicdock.search.plug import make_plugs
from sicdock.data import datadir
from sicdock.util import load, Bunch


def TMP_test_plug(plug, hole):
    print("foo")
    make_plugs(None, None)


def main():
    plug = HC(sicdock.body.Body, datadir + "/pdb/DHR14.pdb")
    hole = HC(sicdock.body.Body, datadir + "/pdb/hole_C3_i52_Z_asym.pdb", sym=3)
    hscore_tables = HC(load_medium_hscore)
    hscore = HierScore(hscore_tables)

    TMP_test_plug(plug, hole)


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
    files = []
    for f in fnames:
        file = "/home/sheffler/debug/sicdock/respairdat/hscore/" + f
        print("=" * 80)
        print("LOADING", file)
        print("=" * 80)
        files.append(load(file))
    return files


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
    files = []
    for f in fnames:
        file = "/home/sheffler/debug/sicdock/respairdat/hscore/" + f
        print("=" * 80)
        print("LOADING", file)
        print("=" * 80)
        files.append(load(file))
    return files


if __name__ == "__main__":
    main()
