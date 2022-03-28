import rmsd
import numpy as np
import rpxdock as rp
import willutil as wu
from willutil.homog import htrans, hrot, hxform

class DeathStar(object):
   """represents data for asymmetrized cage"""
   def __init__(
         self,
         hull,
         laser,
         cagesym,
         cycsym,
         origin=np.eye(4),
         contact_dist=6,
         clash_dist=3,
   ):
      super(DeathStar, self).__init__()
      self.hull = hull
      self.laser = laser
      self.cagesym = cagesym
      self.cycsym = cycsym
      self.origin = origin

      foo = cage_to_cyclic(hull, cagesym, cycsym, origin)
      self.frames = foo.frames
      self.neighbors = foo.neighbors
      self.nbrs_internal = foo.nbrs_internal
      self.follows = foo.follows
      self.asymunit = foo.asymunit
      self.asymframes = self.frames[self.asymunit]

      self.topid = 0
      self.bottomid = len(self.frames) - 1
      self.dofids = np.arange(len(self.frames) - 2) + 1

      self.symx = wu.sym.frames(cycsym)

      self.xorig1to0 = wu.hinv(self.frames[1]) @ self.frames[0]
      self.set_dofs(self.frames)

      self.contact_dist = contact_dist
      self.clash_dist = clash_dist

      self.__ctact = 0

   def body(self, isub):
      return self.laser if isub == 0 else self.hull

   # def iface_xforms(self):
   # ifaces = list()
   # for (icage, jcage), (icyc, jcyc) in zip(self.neighbors, self.nbrs_internal):
   # ix = self.body(icage).symcom(self.frames[icage])[0, icyc]
   # jx = self.body(jcage).symcom(self.frames[jcage])[0, jcyc]
   # ifaces.append(wu.hinv(ix) @ jx)
   # ifaces = np.array(ifaces)
   # return ifaces

   def dofs(self):
      return self.frames

   def set_dofs(self, frames):
      assert self.dofs_are_valid(frames)
      self.frames[1:] = frames[1:]
      inbr = 1
      xint = wu.hrot([0, 0, 1], 0)
      xnbr = wu.hinv(self.frames[self.neighbors[inbr, 0]]) @ self.frames[self.neighbors[inbr, 1]]
      # xnbr =  self.xorig1to0
      self.frames[0] = self.frames[1] @ xnbr @ xint
      # self.frames[0] = np.eye(4)
      self.symmetrize()

   def ncontacts(self, begnbr=1):
      return int(
         np.mean(
            self.hull.contact_count(
               self.hull,
               self.frames[self.neighbors[begnbr:, 0]],
               self.frames[self.neighbors[begnbr:, 1]],
               self.contact_dist,
            )))

   def iface_score(self, tether=True, timer=None, begnbr=1):
      lever = self.hull.radius_xy_max() * .5
      # xiface = self.iface_xforms()
      # x = xiface[1:]

      clash_counts = self.hull.contact_count_bb(
         self.hull,
         self.frames[self.neighbors[begnbr:, 0]],
         self.frames[self.neighbors[begnbr:, 1]],
         self.clash_dist,
      )
      clash = np.mean(clash_counts)
      if clash > 0 and tether: return 9e9
      # print(clash_counts, )
      # assert 0
      contact_counts = self.hull.contact_count(
         self.hull,
         self.frames[self.neighbors[begnbr:, 0]],
         self.frames[self.neighbors[begnbr:, 1]],
         self.contact_dist,
      )
      ctact = np.mean(contact_counts)
      if ctact < 10 and tether: return 9e9

      score = 0
      if tether:
         # score = score**2 / 2
         # print(self.laser.symcom(self.frames[0]).shape)
         # rimcom = self.laser.symcom(self.frames[1])[0, 1, :2, 3]
         # rimcom = self.laser.symcom(self.frames[0])[0, 2, :2, 3]
         spread = self.getspread()
         # score += -spread / 1
         # print(wu.hnorm(rimcom))
         # assert 0
         # score += (49.58 - wu.hnorm(rimcom)) / 3

         # score -= np.sqrt(np.sum(self.frames[0, :2, 3]**2)) / 3
         # score += 0.001 * ((50 - ctact) / 10)**2
         # score += clash * 100000

      # score += wu.hcoherence(xiface[1:], lever)
      # ref_iface_idx = np.argmax(contact_counts)
      ref_iface_idx = -1
      rmsds, fitted = self.iface_joint_rmsd(ref_iface_idx=ref_iface_idx, begnbr=begnbr)
      # print(rmsds)
      # score += max(0, max(rmsds) - 0.5)**2

      if not tether:
         return np.max(rmsds)

      # print(spread, np.max(rmsds))
      return -spread / 2 + max(0, np.max(rmsds - 0.5))**2 + 100 * clash - ctact / 10

   def iface_positions(self, ref_iface_idx, begnbr, pairs=None):
      nbrs = self.neighbors[begnbr:]
      nbrsint = self.nbrs_internal[begnbr:]
      if pairs is None:
         xa0 = self.frames[nbrs[ref_iface_idx, 0]]
         xb0 = self.frames[nbrs[ref_iface_idx, 1]]
         pairs = self.hull.contact_pairs(self.hull, xa0, xb0, maxdis=self.contact_dist)

      x = list()
      for inb, (nbr1, nbr2) in enumerate(nbrs):
         if inb == ref_iface_idx: continue
         pos1 = self.frames[nbr1]
         pos2 = self.frames[nbr2]
         coord1 = self.hull.coord[pairs[0, 0]].reshape(-1, 4)
         coord2 = self.hull.coord[pairs[0, 1]].reshape(-1, 4)
         com1 = np.mean(coord1, axis=0)
         com2 = np.mean(coord2, axis=0)
         symcom1 = self.hull.symcom(pos1)[0, nbrsint[inb, 0], :, 3]
         symcom2 = self.hull.symcom(pos2)[0, nbrsint[inb, 1], :, 3]
         # print((wu.hnorm(symcom1 - wu.hxform(self.symx, com1))))
         comdelta1 = wu.hnorm(symcom1 - wu.hxform(pos1, wu.hxform(self.symx, com1)))
         comdelta2 = wu.hnorm(symcom2 - wu.hxform(pos2, wu.hxform(self.symx, com2)))
         # print(comdelta1, comdelta2)
         iint1 = np.argmin(comdelta1)
         iint2 = np.argmin(comdelta2)
         # print(iint1, iint2)
         x.append([pos1 @ self.symx[iint1], pos2 @ self.symx[iint2]])
      return np.array(x)

   def getspread(self):
      return np.linalg.norm(self.frames[0, :2, 3])

   def iface_joint_rmsd(self, ref_iface_idx, begnbr):
      nbrs = self.neighbors[begnbr:]
      nbrsint = self.nbrs_internal[begnbr:]
      xa0 = self.frames[nbrs[ref_iface_idx, 0]]
      xb0 = self.frames[nbrs[ref_iface_idx, 1]]
      pairs = self.hull.contact_pairs(self.hull, xa0, xb0, maxdis=self.contact_dist)
      A = np.concatenate([
         wu.hxform(xa0, self.hull.coord[pairs[:, 0]].reshape(-1, 4)),
         wu.hxform(xb0, self.hull.coord[pairs[:, 1]].reshape(-1, 4)),
      ])
      Acom = np.mean(A, axis=0)
      Acen = wu.hpoint(A[:, :3] - Acom[:3])
      # wu.showme(A, 'A')
      # wu.showme(wu.hpoint(A[:, :3] - Acom[:3]), 'Acen')
      # print(nbrs[ref_iface_idx], nbrsint[ref_iface_idx])
      fitted = list()
      rmsds = list()

      ifacepos = self.iface_positions(ref_iface_idx, begnbr, pairs)

      for inbr, (xa, xb) in enumerate(ifacepos):
         coord1 = wu.hxform(xa, self.hull.coord[pairs[:, 0]].reshape(-1, 4))
         coord2 = wu.hxform(xb, self.hull.coord[pairs[:, 1]].reshape(-1, 4))

         B = np.concatenate([coord1, coord2])
         Bcom = np.mean(B, axis=0)

         # wu.showme(B, 'B')

         # print(A[0])
         Bcen = wu.hpoint(B[:, :3] - Bcom[:3])
         # print(Acen[:3])
         U = rmsd.kabsch(Bcen[:, :3], Acen[:, :3])
         X = np.eye(4)
         X[:3, :3] = U.T
         X = wu.htrans(Acom) @ X @ wu.htrans(-Bcom)
         Bfit = wu.hpoint(wu.hxform(X, B))
         rms = rmsd.rmsd(A, Bfit)
         rmsds.append(rms)
         # wu.showme(Bfit, name='Bfit')
         fitted.append(Bfit)
      rmsds = np.array(rmsds)
      fitted = np.stack(fitted)
      return rmsds, fitted

   def scoredofs(self, dofs, tether=True, timer=None):
      old = self.dofs()
      self.set_dofs(dofs)
      score = self.iface_score(tether)
      self.set_dofs(old)
      return score

   def symmetrize(self):
      for i, j in self.follows.items():
         self.frames[i] = self.symx[1] @ self.frames[j]

   def dofs_are_valid(self, frames):
      return True

def cage_to_cyclic(
      hull,
      cagesym,
      cycsym,
      origin=np.eye(4),
):
   kw = wu.Bunch(headless=True)
   kw.headless = False

   sympos1 = wu.sym.frames(cagesym, axis=[0, 0, 1], asym_of=cycsym, bbsym=cycsym)
   sympos2 = wu.sym.frames(cagesym, axis=[0, 0, 1], asym_of=cycsym, bbsym=cycsym, asym_index=1)
   sub1 = np.linalg.inv(sympos1[0])
   pos1 = sympos1 @ sub1 @ origin
   pos2 = sympos2 @ sub1 @ origin

   pos1 = sort_frames_z(pos1, hull.com())
   pos2 = sort_frames_z(pos2, hull.com())
   flb, fub, fnum = wu.sym.symunit_bounds(cagesym, cycsym)
   pos1b = pos1[flb:fub]

   asymunit = np.arange(len(pos1b), dtype='i')
   follows, pos2 = get_followers_z(cycsym, pos1b, pos2)
   nbrs, nbrs_internal = get_symcom_neighbors(cagesym, cycsym, hull, pos1, pos2)

   pos = np.concatenate([pos1b, pos2])
   pos, nbrs, follows, asymunit = prune_lone_frames(pos, nbrs, follows, asymunit)
   pos, nbrs, follows, asymunit = sort_frames_z(pos, hull.com(), nbrs, follows, asymunit)
   order = np.argsort(nbrs[:, 1])
   nbrs = nbrs[order]
   nbrs_internal = nbrs_internal[order]

   # wu.showme(hull, name='test0', pos=pos, delprev=False, hideprev=False, linewidth=5,
   # col='rand', nbrs=nbrs, **kw)

   return wu.Bunch(
      frames=pos,
      neighbors=nbrs,
      nbrs_internal=nbrs_internal,
      follows=follows,
      asymunit=asymunit,
   )

def get_symcom_neighbors(cagesym, cycsym, hull, pos1, pos2):
   flb, fub, fnum = wu.sym.symunit_bounds(cagesym, cycsym)
   symcom = hull.symcom(pos1)
   comdist1 = hull.symcomdist(pos1[flb:fub], mask=True)
   comdist2 = hull.symcomdist(pos1[flb:fub], pos2)
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
   order = np.argsort(nbrs[:, 1])
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
