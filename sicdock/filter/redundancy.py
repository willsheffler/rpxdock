import numpy as np
from sicdock.cluster import cookie_cutter
from sicdock.util import Bunch

def filter_redundancy(xforms, body, scores, **kw):
   args = Bunch(kw)
   if len(scores) == 0: return []
   nclust = args.max_cluster
   if nclust is None:
      nclust = int(args.beam_size) // 10

   ibest = np.argsort(-scores)
   crd = xforms[ibest[:nclust], None] @ body.cen[::10, :, None]
   ncen = crd.shape[1]
   crd = crd.reshape(-1, 4 * ncen)
   keep = cookie_cutter(crd, args.rmscut * np.sqrt(ncen))
   return ibest[keep]
