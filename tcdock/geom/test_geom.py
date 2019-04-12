from cppimport import import_hook
from . import primitive_test


def test_geom():
    assert primitive_test.test_geom_primitive_sphere()
    assert primitive_test.test_geom_primitive_welzl_bounding_sphere()
