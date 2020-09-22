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

def grid_sym_multi_axis(
        spec,
        cart_bounds=[25,200],
        resl=1,
        angresl=1, 
        flip=True,
        **kw
):
   if not (hasattr(spec, 'nfold') and hasattr(spec, 'axis') and hasattr(spec, 'xflip')):
      raise ValueError('spec must have nfold, axis and xflip')
   assert len(spec.nfold) == len(spec.axis) == len(spec.xflip)

   if isinstance(flip_components, bool):
      flip_components = [flip_components]
   if len(flip_components) is 1:
      flip_components = flip_components * len(spec.nfold)
   
   #TODO: Quinton: Make sure this is correct, I don't think it applies for grid
   if len(cart_bounds) is 2 and isinstance(cart_bounds[0], int):
      cart_bounds = np.array([cart_bounds] * spec.num_components)
   cart_bounds = np.array(cart_bounds)
   assert len(cart_bounds) in (1, len(spec.nfold))
   #cart_bounds = np.tile(cart_bounds, [8, 1])  # for wrapping / repeating

   #TODO: Quinton: This is from the one-comp grid sampler, harmonize with the above to make sure it all behaves.
   """
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
   """
   cart = np.arange(cart_bounds[0], cart_bounds[1], resl)
   
   ang = 360 / spec.nfold
   ang_nstep = np.ceil(ang / angresl).astype('i')

   #TODO: Make everything from here down work for two comp stuff.
   axis=spec.axis
   axis = np.array(axis[:3] / np.linalg.norm(axis[:3]))
   c = hm.htrans(cart[:, None] * axis[:3])
   r = hm.hrot(axis, ang)
   grid = (r[None] @ c[:, None]).reshape(-1, 4, 4)
   if flip:
      cen = np.mean(cart)
      xflip = hm.hrot(flip, 180.0, axis * cen)
      grid = np.concatenate([grid, xflip @ grid])
   return grid
