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
      self.follows = foo.follows
      self.asymunit = foo.asymunit

      self.asymframes = self.frames[self.asymunit]

   def get_dof_xforms():
      pass

def cage_to_cyclic(
      body,
      cagesym,
      cycsym,
      origin=np.eye(4),
):
   kw = wu.Bunch(headless=True)
   kw.headless = False

   sympos1 = wu.sym.frames(cagesym, axis=[0, 0, 1], asym_of=cycsym, bbsym=cycsym)
   sympos2 = wu.sym.frames(cagesym, axis=[0, 0, 1], asym_of=cycsym, bbsym=cycsym, asym_index=1)
   allframes = wu.sym.frames(cagesym, axis=[0, 0, 1], bbsym=cycsym)
   allframes = allframes @ np.linalg.inv(sympos1[0]) @ origin

   pos1 = sympos1 @ np.linalg.inv(sympos1[0]) @ origin
   pos2 = sympos2 @ np.linalg.inv(sympos2[0]) @ origin
   pos1 = sort_frames_z(pos1, body.com())
   pos2 = sort_frames_z(pos2, body.com())
   flb, fub, fnum = wu.sym.symunit_bounds(cagesym, cycsym)
   pos1b = pos1[flb:fub]

   asymunit = np.arange(len(pos1b), dtype='i')
   follows, pos2 = get_followers_z(cycsym, pos1b, pos2)
   nbrs, nbrs_internal = get_symcom_neighbors(cagesym, cycsym, body, pos1, pos2)

   pos = np.concatenate([pos1b, pos2])
   pos, nbrs, follows, asymunit = prune_lone_frames(pos, nbrs, follows, asymunit)
   pos, nbrs, follows, asymunit = sort_frames_z(pos, body.com(), nbrs, follows, asymunit)

   # wu.viz.showme(body, name='test0', pos=pos, delprev=False, hideprev=False, linewidth=5,
   # col='rand', nbrs=nbrs, **kw)

   return wu.Bunch(
      frames=pos,
      allframes=allframes,
      neighbors=nbrs,
      nbrs_internal=nbrs_internal,
      follows=follows,
      asymunit=asymunit,
   )

def get_symcom_neighbors(cagesym, cycsym, body, pos1, pos2):
   flb, fub, fnum = wu.sym.symunit_bounds(cagesym, cycsym)

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
   nbrs_internal = np.stack([np.concatenate([c1, d1]), np.concatenate([c2, d2])]).T
   return nbrs, nbrs_internal

def sort_frames_z(frames, com, nbrs=None, follows=None, asymunit=None):
   # print(nbrs.shape, nbrs_internal.shape)
   order = np.argsort(-(frames @ com)[:, 2])
   inv = order.copy()
   inv[order] = np.arange(len(order))
   frames = frames[order]
   if nbrs is None:
      return frames
   nbrs = inv[nbrs]
   follows = {inv[i]: inv[j] for i, j in follows.items()}
   asymunit = inv[asymunit]
   return frames, nbrs, follows, asymunit

def prune_lone_frames(frames, nbrs, follows=None, asymunit=None):
   # prune
   oldid = np.array(sorted(set(nbrs.flat)))
   newid = -np.ones(max(oldid) + 1, dtype='i')
   newid[oldid] = np.arange(len(oldid))
   frames = frames[oldid]
   nbrs = newid[nbrs]
   follows = {newid[i]: newid[j] for i, j in follows.items() if newid[i] > 0}
   asymunit = newid[asymunit]
   return frames, nbrs, follows, asymunit

def get_followers_z(cycsym, pos1, pos2):
   follows = dict()
   nfold = int(cycsym[1:])
   symx = wu.hrot([0, 0, 1], np.arange(nfold) / nfold * np.pi * 2)
   # for i, x1 in enumerate(pos1):
   #    for j, x2 in enumerate(pos2):
   #       for k, symx2 in enumerate(symx):
   #          if np.allclose(symx[1] @ x1 @ symx2, x2):
   #             follows[i] = len(pos1) + j
   newpos2 = pos2.copy()
   for i, x in enumerate(pos1):
      newpos2[i + 1] = symx[1] @ x
      follows[i + 1 + len(pos1)] = i
   return follows, newpos2
