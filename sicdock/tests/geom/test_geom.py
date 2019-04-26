from cppimport import import_hook
from sicdock.geom import primitive_test


def test_geom():
    assert primitive_test.TEST_geom_primitive_sphere()
    assert primitive_test.TEST_geom_primitive_welzl_bounding_sphere()
