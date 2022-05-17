import pytest, os, sys, _pickle
from os.path import join, dirname, abspath, exists

from rpxdock import data
from rpxdock.rosetta.triggers_init import get_pose_cached
from rpxdock import ResPairData, RpxHier
from rpxdock.search.result import dummy_result

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
   print('fixture datadir', data.datadir)
   assert exists(data.datadir)
   return data.datadir

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
   return data.small_respairdat()

@pytest.fixture(scope="session")
def respairscore(datadir):
   return data.small_respairscore()

@pytest.fixture(scope="session")
def hscore():
   return data.small_hscore()

@pytest.fixture(scope="session")
def body():
   return data.get_body('DHR14')

@pytest.fixture(scope="session")
def body2():
   return data.get_body('top7')

@pytest.fixture(scope="session")
def plug():
   return data.get_body('dhr64')

@pytest.fixture(scope="session")
def hole():
   return data.get_body('small_c3_hole_sym3')

@pytest.fixture(scope="session")
def body_c3_mono():
   return data.get_body('test_c3_mono')

@pytest.fixture(scope="session")
def body_cageA():
   return data.get_body('T33_dn2_asymA')

@pytest.fixture(scope="session")
def body_cageB():
   return data.get_body('T33_dn2_asymB')

@pytest.fixture(scope="session")
def body_cageA_extended():
   return data.get_body('T33_dn2_asymA_extended')

@pytest.fixture(scope="session")
def body_cageB_extended():
   return data.get_body('T33_dn2_asymB_extended')

@pytest.fixture(scope="session")
def bodyC2():
   return data.get_body('C2_REFS10_1')

@pytest.fixture(scope="session")
def bodyC3():
   return data.get_body('C3_1na0-1_1')

@pytest.fixture(scope="session")
def bodyC4():
   return data.get_body('C4_1na0-G1_1')

@pytest.fixture(scope="session")
def bodyC6():
   return data.get_body('C6_3H22')

@pytest.fixture(scope="session")
def body_tiny():
   return data.get_body('tiny')

@pytest.fixture(scope="session")
def result():
   return dummy_result()

@pytest.fixture(scope="session")
def twocomp_result():
   return data.twocomp_result()
