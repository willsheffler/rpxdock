import logging, numpy as np, rpxdock as rp

log = logging.getLogger(__name__)

def filter_redundancy(xforms, body, scores, **kw):
   arg = rp.Bunch(kw)
   if len(scores) == 0: return []

   ibest = np.argsort(-scores)
   if arg.max_bb_redundancy <= 0:
      return ibest

   nclust = arg.max_cluster if arg.max_cluster else int(arg.beam_size) // 10

   if xforms.ndim == 3:
      crd = xforms[ibest[:nclust], None] @ body.cen[::10, :, None]
   else:
      crd0 = xforms[ibest[:nclust], 0, None] @ body[0].cen[::10, :, None]
      crd1 = xforms[ibest[:nclust], 1, None] @ body[1].cen[::10, :, None]
      crd = np.concatenate([crd0, crd1])

   ncen = crd.shape[1]
   crd = crd.reshape(-1, 4 * ncen)

   keep = rp.cluster.cookie_cutter(crd, arg.max_bb_redundancy * np.sqrt(ncen))
   assert len(np.unique(keep)) == len(keep)

   log.info(f'filter_redundancy {arg.max_bb_redundancy}A Nmax {nclust} ' +
            f'Ntotal {len(ibest)} Nkeep {len(keep)}')

   return ibest[keep]
