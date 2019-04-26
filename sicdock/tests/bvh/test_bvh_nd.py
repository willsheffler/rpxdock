import numpy as np
from cppimport import import_hook
from sicdock.bvh import bvh_nd

def test_bvh_nd():
    pts = np.random.randn(10_000,7)
    bvh = bvh_nd.create_bvh7(pts)
    assert np.allclose(bvh.com(), np.mean(pts, axis=0))


if __name__ == '__main__':
    test_bvh_nd()