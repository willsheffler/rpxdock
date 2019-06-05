import numpy as np
import sicdock.homog as hm

def grid_sym_axis(cart, ang, axis=[0, 0, 1]):
   c = hm.htrans(cart[:, None] * np.array(axis[:3]))
   r = hm.hrot(axis, ang)
   grid = r[None] @ c[:, None]
   return grid.reshape(-1, 4, 4)
