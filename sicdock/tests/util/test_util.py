import _pickle
from cppimport import import_hook

from sicdock.util.dilated_int_test import *
from sicdock.util import Bunch


def test_dilated_int():
    assert TEST_dilated_int_64bit()


def test_bunch(tmpdir):
    x = Bunch(dict(a=2, b="bee"))
    x.c = "see"
    with open(tmpdir + "/foo", "wb") as out:
        _pickle.dump(x, out)

    with open(tmpdir + "/foo", "rb") as inp:
        y = _pickle.load(inp)

    assert x == y
    assert y.a == 2
    assert y.b == "bee"
    assert y.c == "see"
