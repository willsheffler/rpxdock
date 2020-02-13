import logging, numpy as np, rpxdock as rp

log = logging.getLogger(__name__)

def filter_redundancy(xforms, body, scores=None, categories=None, every_nth=10, **kw):
   arg = rp.Bunch(kw)
   if scores is None:
      scores = np.repeat(0, len(xforms))
   if len(scores) == 0: return []

   if categories is None:
      categories = np.repeat(0, len(scores))

   ibest = np.argsort(-scores)
   if arg.max_bb_redundancy <= 0:
      return ibest

   nclust = arg.max_cluster if arg.max_cluster else int(arg.beam_size) // every_nth

   if xforms.ndim == 3:
      crd = xforms[ibest[:nclust], None] @ body.cen[::every_nth, :, None]
   else:
      crd0 = xforms[ibest[:nclust], 0, None] @ body[0].cen[::every_nth, :, None]
      crd1 = xforms[ibest[:nclust], 1, None] @ body[1].cen[::every_nth, :, None]
      crd = np.concatenate([crd0, crd1], axis=1)

   ncen = crd.shape[1]
   crd = crd.reshape(-1, 4 * ncen)

   # sneaky way to do categories
   crd += (categories[ibest[:nclust]] * 1_000_000)[:, None]

   keep = rp.cluster.cookie_cutter(crd, arg.max_bb_redundancy * np.sqrt(ncen))
   assert len(np.unique(keep)) == len(keep)

   log.info(f'filter_redundancy {arg.max_bb_redundancy}A Nmax {nclust} ' +
            f'Ntotal {len(ibest)} Nkeep {len(keep)}')

   return ibest[keep]
