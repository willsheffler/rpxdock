from cppimport import import_hook
from . import bvh_test


def test_geom():
    assert bvh_test.test_bvh_test_min()
    assert bvh_test.test_bvh_test_isect()
