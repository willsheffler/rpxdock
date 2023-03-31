import logging, numpy as np, rpxdock as rp
import willutil as wu

log = logging.getLogger(__name__)

def filter_redundancy(xforms, body, scores=None, categories=None, every_nth=10, symframes=None, **kw):
   kw = wu.Bunch(kw, _strict=False)
   if scores is None:
      scores = np.repeat(0, len(xforms))
   if len(scores) == 0: return []

   assert not categories
   if categories is None:
      categories = np.repeat(0, len(scores))
   if isinstance(symframes, str):
      try:
         symframes = wu.sym.frames(symframes)
      except (KeyError, ValueError):
         symframes = None
   if symframes is None:
      symframes = np.array([np.eye(4)])

   nclust = kw.max_cluster if kw.max_cluster else int(kw.beam_size) // every_nth
   nclust = min(nclust, len(xforms))
   ibest = np.argsort(-scores)[:nclust]
   # ic(ibest.shape, ibest.dtype)
   # ic(categories.shape)
   # categories = categories[ibest]
   # ic(categories.shape)

   if kw.max_bb_redundancy <= 0:
      return ibest

   if xforms.ndim == 3:
      crd = wu.hxform(xforms[ibest], body.cen[::every_nth])
   else:
      com0 = wu.hxform(xforms[ibest, 0], body[0].com(), is_points=True, outerprod=True)
      com1 = wu.hxform(xforms[ibest, 1], body[1].com(), is_points=True, outerprod=True)
      com1sym = wu.hxform(symframes, com1)
      dist = np.linalg.norm(com0 - com1sym, axis=-1)
      isym = np.argmin(dist, axis=0)

      if len(isym) < 100:
         pass
         # ic(com0[0])
         # ic(com1[0])
         # ic(com1sym[0,0])
         # ic(com1sym[isym[0],0])
         # ic(wu.hxform(symframes[isym], com1)[0])
      #    ic(dist[:,0])
      #    ic(dist[isym[0],0])

      #    ic(com1sym.shape)
      #    ic(com1sym[:,0])
      #    ic(dist[:,0])

      #    ic(isym.shape)
      #    ic(dist.shape)
      #    ic(dist[isym,0])
      #    ic(np.min(dist,axis=0))
      #    ic(symframes.shape)
      #    ic(symframes[isym].shape)
      #    # ic(isym)
      #    assert 0
      bodycrd0 = body[0].cen[::every_nth]
      bodycrd1 = body[1].cen[::every_nth]
      # bodycrd1 = wu.hxform(symframes[isym], bodycrd1)

      crd0 = wu.hxform(xforms[ibest, 0], bodycrd0, is_points=True, outerprod=False)
      crd1 = wu.hxform(xforms[ibest, 1], bodycrd1, is_points=True, outerprod=False)
      crd1 = wu.hxform(symframes[isym], crd1, is_points=True, outerprod=False)
      crd = np.concatenate([crd0, crd1], axis=1)

   if crd.ndim == 2:
      return ibest

   ncen = crd.shape[1]
   crd = crd.reshape(-1, 4 * ncen)

   # sneaky way to do categories
   # crd += categories[:,None] * 1_000_000

   keep, clustid = rp.cluster.cookie_cutter(crd, kw.max_bb_redundancy * np.sqrt(ncen))
   assert len(np.unique(keep)) == len(keep)

   # log.info(f'filter_redundancy {kw.max_bb_redundancy}A Nmax {nclust} ' + f'Ntotal {len(ibest)} Nkeep {len(keep)}')

   return ibest[keep]
