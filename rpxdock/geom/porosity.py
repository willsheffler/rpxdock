import numpy as np
from rpxdock.geom import symframes
from rpxdock.sampling.sphere import get_sphere_samples

def sphere_porosity(ca, sym, **kw):

   frames = symframes(sym)
   symca = frames[:, None] @ ca
   symca = symca[:, :, :3].squeeze().reshape(-1, 3)
   normca = symca / np.linalg.norm(symca, axis=1)[:, None]
   # compute fraction of sph points overlapped by protein
   # sph is ~1degree covering radius grid
   sph = get_sphere_samples(sym=crit.symname)
   d2 = np.sum((sph[:, None] - normca)**2, axis=2)
   md2 = np.min(d2, axis=1)
   sphere_surface = 4 * np.pi * radius[i]**2
   return np.sum(md2 > 0.002) / len(sph) * sphere_surface
