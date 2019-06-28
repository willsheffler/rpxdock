import rpxdock as rp, numpy as np

def hier_axis_sampler(nfold, lb=25, ub=200, resl=10, angresl=10, axis=[0, 0, 1], flipax=[0, 1,
                                                                                         0]):
   cart_nstep = int(np.ceil((ub - lb) / resl))
   ang = 360 / nfold
   ang_nstep = int(np.ceil(ang / angresl))
   samp = rp.sampling.RotCart1Hier_f4(lb, ub, cart_nstep, 0, ang, ang_nstep, axis[:3])
   if flipax is not None:
      flip = rp.GridHier([np.eye(4), rp.homog.hrot(flipax, 180)])
      samp = rp.ProductHier(samp, flip)
   return samp

def hier_multi_axis_sampler(nfold, axis, xflip, lb=25, ub=200, resl=10, angresl=10, flip=True):
   assert len(nfold) == len(axis) == len(xflip)
   cart_nstep = int(np.ceil((ub - lb) / resl))
   ang = 360 / nfold
   ang_nstep = np.ceil(ang / angresl).astype('i')

   samp = [
      rp.sampling.RotCart1Hier_f4(lb, ub, cart_nstep, 0, ang[i], ang_nstep[i], axis[i][:3])
      for i in range(len(nfold))
   ]
   if isinstance(flip, bool): flip = [flip] * len(nfold)
   for i, s in enumerate(samp):
      if flip[i]: samp[i] = rp.ProductHier(s, rp.GridHier([np.eye(4), xflip[i]]))

   return rp.CompoundHier(*samp)
