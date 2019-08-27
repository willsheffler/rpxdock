import rpxdock as rp, numpy as np

class LineHier:
   def __init__(self, lb, ub, nstep, axis):
      self.hier1d = rp.sampling.CartHier1D_f4([lb], [ub], [nstep])
      self.axis = rp.homog.hnormalized(axis).squeeze()[:3]
      self.ncell = nstep
      self.dim = 1
      self.state = lb, ub, nstep, self.axis

   def get_xforms(self, resl, idx):
      ok, disp = self.hier1d.get_trans(resl, idx)
      return ok, rp.homog.htrans(self.axis * disp[:, 0, None])

   def __getstate__(self):
      return self.state

   def __setstate__(self, state):
      self.axis = state[3]
      self.hier1d = rp.sampling.CartHier1D_f4([state[0]], [state[1]], [state[2]])

def hier_axis_sampler(nfold, lb=25, ub=200, resl=10, angresl=10, axis=[0, 0, 1], flipax=[0, 1,
                                                                                         0]):
   cart_nstep = int(np.ceil((ub - lb) / resl))
   ang = 360 / nfold
   ang_nstep = int(np.ceil(ang / angresl))
   samp = rp.sampling.RotCart1Hier_f4(lb, ub, cart_nstep, 0, ang, ang_nstep, axis[:3])
   if flipax is not None:
      flip = rp.ZeroDHier([np.eye(4), rp.homog.hrot(flipax, 180)])
      samp = rp.ProductHier(samp, flip)
   return samp

def hier_multi_axis_sampler(spec, *args, **kw):
   if spec.spec.startswith('P'):
      return hier_multi_axis_sampler_lattice(spec, *args, **kw)
   else:
      return hier_multi_axis_sampler_cage(spec, *args, **kw)

def hier_multi_axis_sampler_lattice(spec, cart_bounds=[25, 200], resl=10, angresl=10,
                                    flip_components=True, **kw):
   assert 0

def hier_multi_axis_sampler_cage(spec, cart_bounds=[25, 200], resl=10, angresl=10,
                                 flip_components=True, **kw):
   if not (hasattr(spec, 'nfold') and hasattr(spec, 'axis') and hasattr(spec, 'xflip')):
      raise ValueError('spec must have nfold, axis and xflip')
   assert len(spec.nfold) == len(spec.axis) == len(spec.xflip)
   if isinstance(flip_components, bool):
      flip_components = [flip_components]
   if len(flip_components) is 1:
      flip_components = flip_components * len(spec.nfold)
   for i, flip in enumerate(flip_components):
      flip_components[i] = flip_components[i] and not spec.is_dihedral[i]

   cart_nstep = int(np.ceil((cart_bounds[1] - cart_bounds[0]) / resl))
   ang = 360 / spec.nfold
   ang_nstep = np.ceil(ang / angresl).astype('i')

   samp = []
   for i in range(len(spec.nfold)):
      if spec.is_dihedral[i]:
         s = LineHier(cart_bounds[0], cart_bounds[1], cart_nstep, spec.axis[i])
      else:
         s = rp.sampling.RotCart1Hier_f4(cart_bounds[0], cart_bounds[1], cart_nstep, 0, ang[i],
                                         ang_nstep[i], spec.axis[i][:3])
      samp.append(s)

   for i, s in enumerate(samp):
      if flip_components[i]:
         samp[i] = rp.ProductHier(s, rp.ZeroDHier([np.eye(4), spec.xflip[i]]))

   sampler = rp.CompoundHier(*samp)
   sampler.attrs = dict(spec=spec, cart_bounds=cart_bounds, resl=resl, angresl=angresl,
                        flip_components=flip_components)
   return sampler
