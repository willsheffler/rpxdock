from cppimport import import_hook
from sicdock.util.SimpleArray_test import *
from sicdock.util.dilated_int_test import *


def test_SimpleArray():
    assert TEST_SimpleArray_iteration()
    assert TEST_SimpleArray_bounds_check()


def test_dilated_int():
    assert TEST_dilated_int_64bit()


if __name__ == "__main__":
    test_dilated_int()
