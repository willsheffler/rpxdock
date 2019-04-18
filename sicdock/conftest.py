from sicdock.rosetta import get_pose_cached
import pytest
import os
import sys
from os.path import join, dirname, abspath, exists


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
