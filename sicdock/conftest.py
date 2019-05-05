from sicdock.rosetta import get_pose_cached
import pytest, os, sys, _pickle
from os.path import join, dirname, abspath, exists
from sicdock.motif.pairdat import ResPairData
import sicdock

# addoption doesn't work for me
# def pytest_addoption(parser):
#     parser.addoption(
#         "--runslow", action="store_true", default=False, help="run slow tests"
#     )
#
#
# def pytest_collection_modifyitems(config, items):
#     if config.getoption("--runslow"):
#         # --runslow given in cli: do not skip slow tests
#         return
#     skip_slow = pytest.mark.skip(reason="need --runslow option to run")
#     for item in items:
#         if "slow" in item.keywords:
#             item.add_marker(skip_slow)


@pytest.fixture(scope="session")
def datadir():
    root = join(dirname(__file__))
    d = join(root, "data")
    print(d)
    assert exists(d)
    return d


@pytest.fixture(scope="session")
def pdbdir(datadir):
    d = join(datadir, "pdb")
    assert exists(d)
    return d


@pytest.fixture(scope="session")
def C2_3hm4(pdbdir):
    return get_pose_cached("C2_3hm4_1.pdb.gz", pdbdir)


@pytest.fixture(scope="session")
def C3_1nza(pdbdir):
    return get_pose_cached("C3_1nza_1.pdb.gz", pdbdir)


@pytest.fixture(scope="session")
def top7(pdbdir):
    return get_pose_cached("top7.pdb.gz", pdbdir)


@pytest.fixture(scope="session")
def C5_1ojx(pdbdir):
    return get_pose_cached("C5_1ojx.pdb.gz", pdbdir)


@pytest.fixture(scope="session")
def respairdat(datadir):
    with open(join(datadir, "respairdat10_plus_xmap_rots.pickle"), "rb") as inp:
        return ResPairData(_pickle.load(inp))


@pytest.fixture(scope="session")
def respairscore(datadir):
    with open(join(datadir, "pairscore10.pickle"), "rb") as inp:
        return _pickle.load(inp)


@pytest.fixture(scope="session")
def hscore(datadir):
    pref = datadir + "/hscore/"
    return sicdock.motif.hierscore.HierScore(
        [
            pref + "pdb_res_pair_data_si30_10_rots_noSS_p0.5_b1_base.pickle",
            pref + "pdb_res_pair_data_si30_10_rots_noSS_p0.5_b1_hier0_Kflat_1_0.pickle",
            pref + "pdb_res_pair_data_si30_10_rots_noSS_p0.5_b1_hier1_Kflat_1_0.pickle",
            pref + "pdb_res_pair_data_si30_10_rots_noSS_p0.5_b1_hier2_Kflat_1_0.pickle",
            pref + "pdb_res_pair_data_si30_10_rots_noSS_p0.5_b1_hier3_Kflat_1_0.pickle",
            pref + "pdb_res_pair_data_si30_10_rots_noSS_p0.5_b1_hier4_Kflat_1_0.pickle",
        ]
    )
