import rmsd
import numpy as np
import rpxdock as rp
import willutil as wu
from willutil.homog import htrans, hrot, hxform

class DSError(Exception):
   pass

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
         begnbr=1,
         origframes=None,
         capcen=np.eye(4),
         capaln=np.eye(4),
   ):
      super(DeathStar, self).__init__()

      self.cap_xform = wu.htrans([0, 9, 0]) @ wu.hrot([1, 0, 0], -3, degrees=True)

      self.hull = hull
      self.laser = laser
      self.cagesym = cagesym
      self.cycsym = cycsym
      self.origin = origin
      self.begnbr = begnbr
      self.contact_dist = contact_dist
      self.clash_dist = clash_dist
      self.symx = wu.sym.frames(cycsym)
      self.capcen = capcen
      self.capaln = capaln
      self.timer = wu.Timer()

      foo = cage_to_cyclic(hull, cagesym, cycsym, origin)
      self.frames = foo.frames
      self.neighbors = foo.neighbors
      self.nbrs_internal = foo.nbrs_internal
      self.follows = foo.follows
      self.asymunit = foo.asymunit
      # self.asymframes = self.frames[self.asymunit]
      self.ref_iface_idx = len(self.neighbors) - 1 - self.begnbr  # totally arbitrary

      self.origframes = self.frames if origframes is None else origframes
      self.orig_iface_rel_xforms, _ = self.iface_rel_xforms(original=True)
      assert np.allclose(self.orig_iface_rel_xforms, self.orig_iface_rel_xforms[0], atol=1e-4)

      self.topid = 0
      self.bottomid = len(self.frames) - 1
      self.dofids = np.arange(len(self.frames) - 2) + 1

      # self.xorig1to0 = wu.hinv(self.frames[1]) @ self.frames[0]
      self.set_dofs(self.frames)

      self.__ctact = 0

      self.lever = 30

      # ca = self.capaln
      # cc = self.capcen
      # cia = wu.hinv(self.capaln)
      # cic = wu.hinv(self.capcen)

   def get_asym_frames(self):
      return self.frames[self.asymunit]

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

      xint = wu.hrot([0, 0, 1], 240)  # ???
      # np.set_printoptions(precision=5, suppress=True)
      xnbrs = self.iface_positions()
      xnbrs = wu.hinv(xnbrs[1:, 1]) @ xnbrs[1:, 0]
      xnbr = wu.hmean(xnbrs)

      frame0 = self.frames[1] @ xnbr @ self.cap_xform @ xint
      self.frames[0] = frame0
      self.symmetrize()
      self.timer.checkpoint('set_dofs')

   def ncontacts(self):
      return int(
         np.mean(
            self.hull.contact_count(
               self.hull,
               self.frames[self.neighbors[self.begnbr:, 0]],
               self.frames[self.neighbors[self.begnbr:, 1]],
               self.contact_dist,
            )))

   def iface_score(self, fullscore=True, timer=None):
      lever = self.hull.radius_xy_max() * .5
      # xiface = self.iface_xforms()
      # x = xiface[1:]

      # clash_counts = self.hull.contact_count_bb(
      #    self.hull,
      #    self.frames[self.neighbors[self.begnbr:, 0]],
      #    self.frames[self.neighbors[self.begnbr:, 1]],
      #    self.clash_dist,
      # )
      # clash = np.mean(clash_counts)
      # if clash > 0 and fullscore: return 9e9
      clashdist = self.hull.distance_to(
         self.hull,
         self.frames[self.neighbors[self.begnbr:, 0]],
         self.frames[self.neighbors[self.begnbr:, 1]],
      )
      clashdist = np.min(clashdist)
      # print(clashdist, )
      # if clashdist < 2.5 and fullscore: return 9e9

      #

      #contact_counts = self.hull.contact_count(
      #   self.hull,
      #   self.frames[self.neighbors[self.begnbr:, 0]],
      #   self.frames[self.neighbors[self.begnbr:, 1]],
      #   self.contact_dist,
      #)
      #ctact = np.mean(contact_counts)
      #if ctact < 10 and fullscore: return 9e9

      score = 0
      if fullscore:
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
      # self.ref_iface_idx = np.argmax(contact_counts)

      # rmsds, fitted = self.iface_joint_rmsd(report=not fullscore)
      diff, origdiff, ifacesep = self.iface_joint_xform_diff(report=not fullscore)
      # diff = np.max(diffs)
      # print(diffs)
      # score += max(0, max(diffs) - 0.5)**2

      if not fullscore:
         return diff

      clashsc = min(0, 3.8 - clashdist)**4 * 10
      # ctact = 1000
      # ctactsc = -ctact / 1000
      ctactsc = 0

      # origdiff = wu.hdiff(self.origframes[[2]], self.frames[[2]], self.lever)

      # print(origdiff)
      # if not fullscore: print(origdiff, end='')

      # print(spread, np.max(diffs))
      diffwellwidth = 0.0  # pretty loose, rms's around 1.8
      diffscore = max(0, diff - diffwellwidth)**4

      self.timer.checkpoint('iface_score')

      axscore = self.getspread()

      # print(diff, diffscore)
      # print(origdiff)
      # return diffscore

      return sum([
         # -spread,
         10 * axscore,
         1 * diffscore,
         # ctactsc,
         clashsc,
         # -10 * origdiff,
         # -100 * ifacesep,
      ])

   def dump_pdb(self, fprefix):
      # for i in range(0,9): print(cmd.fit('dstar_test_iface%iA and resi -80'%i, 'dstar_test_iface8A and resi -80'))
      asym = self.frames[self.asymunit]
      # fprefix = fprefix.replace('.pdb', '')
      self.laser.pos = self.frames[0]
      rp.io.dump_pdb_from_bodies(fprefix + '_cap.pdb', [self.laser])

      self.hull.pos = self.frames[-1]
      rp.io.dump_pdb_from_bodies(fprefix + '_bottom.pdb', [self.hull])
      for i, f in enumerate(asym):
         for j, x in enumerate(self.symx):
            self.hull.pos = x @ f
            rp.io.dump_pdb_from_bodies(f'{fprefix}_hull{i}{j}.pdb', [self.hull])

      ifacepos = self.iface_positions()
      xaln = np.eye(4)
      for inbr, (pos1, pos2) in enumerate(ifacepos):
         x1 = xaln @ np.eye(4)  #wu.hinv(pos1) @ pos1
         x2 = xaln @ wu.hinv(pos1) @ pos2
         ha, hb = self.hull.copy(), self.hull.copy()
         ha.pos = x1
         hb.pos = x2
         rp.io.dump_pdb_from_bodies(f'{fprefix}_iface{inbr}.pdb', [ha, hb])
      #

      # whitetopn = 1
      # capstop = 0
      # pos = np.concatenate([self.frames[:capstop], pos])
      # pos = wu.hxform(self.symx, pos, outerprod=True).swapaxes(0, 1).reshape(-1, 4, 4)
      # pos = np.concatenate([self.frames[:1 - capstop], pos, self.frames[-1:]])
      #
      # bodies = [self.hull.copy() for i in range(len(pos) - 1)]
      # for i in range(len(pos) - 1):
      #    bodies[i].pos = pos[i + 1]
      # self.laser.pos = pos[0]
      # bodies.append(self.laser)
      # s, _ = rp.io.make_pdb_from_bodies(bodies)
      # assert s
      # with open(fname, 'w') as out:
      #    out.write(s)

   def getspread(self):
      axscore = np.sqrt(np.sum(self.frames[0, :2, 2]**2)) * self.lever
      axscore += np.sqrt(np.sum(self.frames[0, :2, 3]**2))
      # + wu.homog.line_line_distance_pa(
      # [0, 0, 0, 1],
      # [0, 0, 1, 0],
      # self.frames[0, :, 3],
      # self.frames[0, :, 2],
      # )
      return axscore
      # return wu.hdiff(self.frames[0], self.origframes[0], lever=self.lever)

   def iface_positions(self, pairs=None, original=False):
      self.timer.checkpoint('iface_positions_beg')
      frames = self.frames
      if original:
         frames = self.origframes
      nbrs = self.neighbors[self.begnbr:]
      nbrsint = self.nbrs_internal[self.begnbr:]
      if pairs is None:
         xa0 = frames[nbrs[self.ref_iface_idx, 0]]
         xb0 = frames[nbrs[self.ref_iface_idx, 1]]
         pairs = self.hull.contact_pairs(self.hull, xa0, xb0, maxdis=self.contact_dist)
         if len(pairs) == 0: raise DSError
      self.timer.checkpoint('hull.contact_pairs')
      x = list()
      for inb, (nbr1, nbr2) in enumerate(nbrs):
         self.timer.checkpoint('ifp beg')
         pos1 = frames[nbr1]
         pos2 = frames[nbr2]
         # print(inb, nbr1, nbr2, 'wu.hdist(pos1, pos2)', wu.hdist(pos1, pos2))
         coord1 = self.hull.coord[pairs[0, 0]].reshape(-1, 4)
         coord2 = self.hull.coord[pairs[0, 1]].reshape(-1, 4)
         com1 = np.mean(coord1, axis=0)
         com2 = np.mean(coord2, axis=0)
         self.timer.checkpoint('ifp coms')
         symcom1 = self.hull.symcom(pos1)[0, nbrsint[inb, 0], :, 3]
         symcom2 = self.hull.symcom(pos2)[0, nbrsint[inb, 1], :, 3]
         self.timer.checkpoint('ifp symcoms')
         # print((wu.hnorm(symcom1 - wu.hxform(self.symx, com1))))
         # print(symcom1.shape, pos1.shape, com1.shape, self.symx.shape)
         comdelta1 = wu.hnorm(symcom1 - wu.hxform(pos1, wu.hxform(self.symx, com1)))
         comdelta2 = wu.hnorm(symcom2 - wu.hxform(pos2, wu.hxform(self.symx, com2)))
         self.timer.checkpoint('ifp comsdeltas')
         # print(comdelta1, comdelta2)
         iint1 = np.argmin(comdelta1)
         iint2 = np.argmin(comdelta2)
         self.timer.checkpoint('ifp argmin')
         # print(iint1, iint2)
         # for i in range(3):
         #    for j in range(3):
         #       print(i, j, wu.hdist(pos1 @ self.symx[i], pos2 @ self.symx[j]))
         pos1 = pos1 @ self.symx[iint1]
         pos2 = pos2 @ self.symx[iint2]

         # assert wu.hdist(pos1, pos2) < 50
         x.append([pos1, pos2])

      self.timer.checkpoint('iface_positions')
      return np.array(x)

   def iface_rel_xforms(self, original=False):
      self.timer.checkpoint('iface_rel_xforms beg')
      ifacepos = self.iface_positions(original=original)
      xaln = np.eye(4)
      xaln = wu.hinv(ifacepos[self.ref_iface_idx, 0]) @ ifacepos[self.ref_iface_idx, 1]
      xaln = wu.hinv(xaln)
      ifpos = xaln @ wu.hinv(ifacepos[:, 0]) @ ifacepos[:, 1]
      self.timer.checkpoint('iface_rel_xforms')
      return ifpos, xaln

   def iface_joint_xform_diff(self, report=False):
      #ifacepos = self.iface_positions()
      #xaln = np.eye(4)
      #xaln = wu.hinv(ifacepos[self.ref_iface_idx, 0]) @ ifacepos[self.ref_iface_idx, 1]
      ## xaln = wu.hrot([0, 1, 0], 45) @ wu.hinv(xaln)
      ## xaln = wu.hrot([1, 0, 0], 90) @ xaln
      #ifpos = xaln @ wu.hinv(ifacepos[:, 0]) @ ifacepos[:, 1]
      ifpos, _ = self.iface_rel_xforms()
      # ifpos = ifpos[self.begnbr:]

      ifmean = wu.hmean(ifpos)
      # ang = wu.hangle_of(ifmean, ifpos)
      # print(ang, wu.hnorm(ifpos))
      diff = wu.hdiff(ifmean, ifpos, self.lever)
      # wu.PING(diff)
      # if np.random.rand() < 0.001: assert 0
      # origdiff = wu.hdiff(ifmean, self.orig_iface_rel_xforms, self.lever)
      origdiff = wu.hdiff(ifmean, self.orig_iface_rel_xforms[0], self.lever)

      # np.set_printoptions(suppress=True, precision=5)
      # print(len(self.neighbors), np.linalg.norm(ifpos[:, :3, 3], axis=-1))
      # print(ifmean[:3, 3])
      sep = np.linalg.norm(ifmean[:3, 3])
      sep = NotImplemented
      # print(sep)
      # assert sep < 10
      # if np.random.rand() < 0.01: assert 0

      # xdiff = wu.hinv(ifpos) @ ifmean
      # diff = wu.hnorm2(xdiff[:, 3]) + wu.hangle(xdiff)**2 * self.lever**2
      # diff = np.sqrt(diff)
      # if report:
      # print('iface_joint_xform_diff', diff)
      # print('iface_joint_xform_diff', diff)
      # assert 0
      self.timer.checkpoint('iface_joint_xform_diff')
      # print(origdiff)
      return np.max(diff), origdiff, sep

   def iface_joint_rmsd_sketchy(self, report=False):
      raise NotImplementedError
      nbrs = self.neighbors[self.begnbr:]
      nbrsint = self.nbrs_internal[self.begnbr:]
      xa0 = self.frames[nbrs[self.ref_iface_idx, 0]]
      xb0 = self.frames[nbrs[self.ref_iface_idx, 1]]
      pairs = self.hull.contact_pairs(self.hull, xa0, xb0, maxdis=self.contact_dist)
      A = np.concatenate([
         wu.hxform(xa0, self.hull.coord[pairs[:, 0]].reshape(-1, 4)),
         wu.hxform(xb0, self.hull.coord[pairs[:, 1]].reshape(-1, 4)),
      ])
      Acom = np.mean(A, axis=0)
      Acen = wu.hpoint(A[:, :3] - Acom[:3])
      # wu.showme(A, 'A')
      # wu.showme(wu.hpoint(A[:, :3] - Acom[:3]), 'Acen')
      # print(nbrs[self.ref_iface_idx], nbrsint[self.ref_iface_idx])
      fitted = list()
      rmsds = list()

      ifacepos = self.iface_positions(pairs)

      for inbr, (xa, xb) in enumerate(ifacepos):
         coord1 = wu.hxform(xa, self.hull.coord[pairs[:, 0]].reshape(-1, 4))
         coord2 = wu.hxform(xb, self.hull.coord[pairs[:, 1]].reshape(-1, 4))
         B = np.concatenate([coord1, coord2])
         Bcom = np.mean(B, axis=0)
         Bcen = wu.hpoint(B[:, :3] - Bcom[:3])
         U = rmsd.kabsch(Bcen[:, :3], Acen[:, :3])
         X = np.eye(4)
         X[:3, :3] = U.T
         X = wu.htrans(Acom) @ X @ wu.htrans(-Bcom)
         Bfit = wu.hpoint(wu.hxform(X, B))
         rms = rmsd.rmsd(A, Bfit)
         rmsds.append(rms)
         fitted.append(Bfit)
      rmsds = np.array(rmsds)
      if report:
         np.set_printoptions(precision=5, suppress=True)
         # print(f'{np.mean(rmsds):7.3f} {np.max(rmsds):7.3f}', end=' ')
         print('           ', rmsds)
      fitted = np.stack(fitted)
      return rmsds, fitted

   def scoredofs(self, dofs, tether=True, timer=None):
      try:
         old = self.dofs()
         self.set_dofs(dofs)
         score = self.iface_score(tether)
         self.set_dofs(old)
         self.timer.checkpoint('scoredofs')
         return score
      except DSError:
         return 9e9

   def symmetrize(self):
      for i, j in self.follows.items():
         self.frames[i] = self.symx[1] @ self.frames[j]
      self.timer.checkpoint('symmetrize')

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

'''
Times(name=Timer, order=longest, summary=sum):
            ifp comsdeltas  785.20412
      iface_rel_xforms beg  696.42333
                  ifp coms  501.75818
                symmetrize  365.08607
               ifp symcoms  343.06837
                      iter  242.01698
    iface_joint_xform_diff  217.73740
                   purturb  205.78882
                ifp argmin  161.80018
       iface_positions_beg  158.54520
                   ifp beg  158.43132
        hull.contact_pairs   82.02490
          iface_rel_xforms   43.13096
               iface_score   18.17109
           iface_positions   15.49444
                MonteCarlo    7.97622
                  set_dofs    6.68384
                 scoredofs    2.27779
                   trythis    1.96153
          MonteCarlo score    1.51781
                   genrand    0.02778
'''