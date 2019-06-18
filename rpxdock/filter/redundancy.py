import numpy as np
from rpxdock.cluster import cookie_cutter
from rpxdock.util import Bunch

def filter_redundancy(xforms, body, scores, **kw):
   args = Bunch(kw)
   if len(scores) == 0: return []

   ibest = np.argsort(-scores)
   if args.max_bb_redundancy <= 0:
      return ibest

   nclust = args.max_cluster
   if nclust is None:
      nclust = int(args.beam_size) // 10

   crd = xforms[ibest[:nclust], None] @ body.cen[::10, :, None]
   ncen = crd.shape[1]
   crd = crd.reshape(-1, 4 * ncen)

   keep = cookie_cutter(crd, args.max_bb_redundancy * np.sqrt(ncen))
   assert len(np.unique(keep)) == len(keep)

   print('filter_redundancy', len(ibest), len(keep))

   return ibest[keep]
