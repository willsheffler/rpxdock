from sicdock.sampling.xform_grid import grid_sym_axis
import sicdock.homog as hm
import numpy as np

def test_grid_sym_axis():
   cart = np.arange(-10.5, 10.6)
   ang = np.arange(-60, 60.1, 10)
   axis = hm.rand_vec()
   grid = grid_sym_axis(cart=cart, ang=ang, axis=axis)
   assert grid.shape == (len(cart) * len(ang), 4, 4)
   assert 0.0001 > np.max(hm.line_angle(grid[:, :3, 3], axis))

if __name__ == '__main__':
   test_grid_sym_axis()
