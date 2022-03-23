import numpy as np
import rpxdock as rp
import willutil as wu
from willutil.homog import htrans, hrot, hxform

class DeathStar(object):
   """represents data for asymmetrized cage"""
   def __init__(
         self,
         body,
         cap,
         cagesym,
         cycsym,
         origin=np.eye(4),
         ifaceclasses=None,
         dofclasses=None,
         doftypes=None,
   ):
      super(DeathStar, self).__init__()
      self.body = body
      self.cap = cap
      self.cagesym = cagesym
      self.cycsym = cycsym
      self.origin = origin
      foo = cage_to_cyclic(body, cagesym, cycsym, origin)
      self.frames = foo.frames
      self.allframes = foo.allframes
      self.neighbors = foo.neighbors
      self.nbrs_internal = foo.nbrs_internal

      self.ifaceclasses = ifaceclasses
      self.dofclasses = dofclasses
      self.doftypes = doftypes

def cage_to_cyclic(
      body,
      cagesym,
      cycsym,
      origin=np.eye(4),
):
   kw = wu.Bunch(headless=True)
   kw.headless = False

   flb, fub, fnum = 1, -1, 2
   if cagesym == 'tet' and cycsym == 'c3':
      fub, fnum = None, 1

   sympos1 = wu.sym.frames(cagesym, axis=[0, 0, 1], asym_of=cycsym, bbsym=cycsym)

   sympos2 = wu.sym.frames(cagesym, axis=[0, 0, 1], asym_of=cycsym, bbsym=cycsym, asym_index=1)
   pos1 = sympos1 @ np.linalg.inv(sympos1[0]) @ origin
   pos2 = sympos2 @ np.linalg.inv(sympos2[0]) @ origin
   pos = np.concatenate([pos1[flb:fub], pos2])

   allframes = wu.sym.frames(cagesym, axis=[0, 0, 1], bbsym=cycsym)
   allframes = allframes @ np.linalg.inv(sympos1[0]) @ origin

   symcom = body.symcom(pos1)
   comdist1 = body.symcomdist(pos1[flb:fub], mask=True)
   comdist2 = body.symcomdist(pos1[flb:fub], pos2)
   if len(pos1[flb:fub]) > 1:
      # print(pos1.shape, comdist1)
      a1, c1, a2, c2 = np.where(comdist1 < np.min(comdist1) + 0.001)
   else:
      a1, a2 = np.array([], dtype='i'), np.array([], dtype='i')
      c1, c2 = np.array([], dtype='i'), np.array([], dtype='i')
   b1, d1, b2, d2 = np.where(comdist2 < np.min(comdist2) + 0.001)
   # assert 0
   nbrs = np.stack([np.concatenate([a1, b1]), np.concatenate([a2, b2 + len(pos1) - fnum])]).T
   nbrsinternal = np.stack([np.concatenate([c1, d1]), np.concatenate([c2, d2])]).T

   # prune
   oldid = np.array(sorted(set(nbrs.flat)))
   newid = -np.ones(max(oldid) + 1, dtype='i')
   newid[oldid] = np.arange(len(oldid))
   pos = pos[oldid]
   nbrs = newid[nbrs]

   # print(nbrs.shape, nbrsinternal.shape)
   order = np.argsort(-(pos @ body.com())[:, 2])
   inv = order.copy()
   inv[order] = np.arange(len(order))
   pos = pos[order]
   nbrs = inv[nbrs]

   # wu.viz.showme(body, name='test0', pos=pos, delprev=False, hideprev=False, linewidth=5,
   # col='rand', nbrs=nbrs, **kw)

   return wu.Bunch(
      frames=pos,
      allframes=allframes,
      neighbors=nbrs,
      nbrs_internal=nbrsinternal,
   )
