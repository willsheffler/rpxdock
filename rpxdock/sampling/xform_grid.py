import numpy as np
import rpxdock.homog as hm

def grid_sym_axis(cart, ang, axis=[0, 0, 1], flip=None):
   if not isinstance(axis, (np.ndarray, list, tuple)):
      raise TypeError('axis must be ndarray, list tuple')
   if len(axis) not in (3, 4):
      raise ValueError('bad axis')
   if flip is not None and not isinstance(flip, (np.ndarray, list, tuple)):
      raise TypeError('flip axis must be ndarray, list tuple')
   if flip and len(flip) not in (3, 4):
      raise ValueError('bad flip axis')
   if flip and hm.hdot(axis, flip) > 0.001:
      raise ValueError('flip axis must be perpendicular to main axis')
   axis = np.array(axis[:3] / np.linalg.norm(axis[:3]))
   c = hm.htrans(cart[:, None] * axis[:3])
   r = hm.hrot(axis, ang)
   grid = (r[None] @ c[:, None]).reshape(-1, 4, 4)
   if flip:
      cen = np.mean(cart)
      xflip = hm.hrot(flip, 180.0, axis * cen)
      grid = np.concatenate([grid, xflip @ grid])
   return grid
