import copy

import numpy as np
import sicdock.geom.homog as hm

from sicdock import bvh
from sicdock.util.numeric import pca_eig
from sicdock import motif

import sicdock.rosetta as ros

_CLASH_RADIUS = 1.75

class Body:
   def __init__(self, pdb, sym="C1", which_ss="HE", posecache=False, **kw):
      if isinstance(pdb, str):
         self.pdbfile = pdb
         if posecache:
            self.pose = ros.get_pose_cached(pdb)
         else:
            self.pose = ros.pose_from_file(pdb)
            ros.assign_secstruct(self.pose)
      else:
         self.pose = pdb

      if isinstance(sym, int):
         sym = "C%i" % sym
      self.sym = sym
      self.nfold = int(sym[1:])
      self.seq = np.array(list(self.pose.sequence()))
      self.ss = np.array(list(self.pose.secstruct()))
      self.ssid = motif.ss_to_ssid(self.ss)
      self.coord = ros.get_bb_coords(self.pose)
      self.chain = np.repeat(0, self.seq.shape[0])
      self.resno = np.arange(len(self.seq))

      if sym and sym[0] == "C" and int(sym[1:]):
         n = self.coord.shape[0]
         nfold = int(sym[1:])
         self.seq = np.array(list(nfold * self.pose.sequence()))
         self.ss = np.array(list(nfold * self.pose.secstruct()))
         self.ssid = motif.ss_to_ssid(self.ss)
         self.chain = np.repeat(range(nfold), n)
         self.resno = np.tile(range(n), nfold)
         newcoord = np.empty((nfold * n, ) + self.coord.shape[1:])
         newcoord[:n] = self.coord
         # print(self.coord.shape, newcoord.shape)
         for i in range(1, nfold):
            self.pos = hm.hrot([0, 0, 1], 360.0 * i / nfold)
            newcoord[i * n:][:n] = self.positioned_coord()
         self.coord = newcoord
      else:
         raise ValueError("unknown symmetry: " + sym)
      assert len(self.seq) == len(self.coord)
      assert len(self.ss) == len(self.coord)
      assert len(self.chain) == len(self.coord)

      self.nres = len(self.coord)
      self.stub = motif.bb_stubs(self.coord)
      self.bvh_bb = bvh.bvh_create(self.coord[..., :3].reshape(-1, 3))
      self.allcen = self.stub[:, :, 3]
      which_cen = np.repeat(False, len(self.ss))
      for ss in "EHL":
         if ss in which_ss:
            which_cen |= self.ss == ss
      which_cen &= ~np.isin(self.seq, ["G", "C", "P"])
      self.which_cen = which_cen
      self.bvh_cen = bvh.bvh_create(self.allcen[:, :3], which_cen)
      self.cen = self.allcen[which_cen]
      self.pos = np.eye(4, dtype="f4")
      self.pair_buf = np.empty((10000, 2), dtype="i4")
      self.pcavals, self.pcavecs = pca_eig(self.cen)

      if self.sym != "C1":
         self.asym_body = Body(pdb, "C1", which_ss, posecache, **kw)
      else:
         self.asym_body = self

   def com(self):
      return self.pos @ self.bvh_bb.com()

   def rg(self):
      d = self.cen - self.com()
      return np.sqrt(np.sum(d**2) / len(d))

   def radius_max(self):
      return np.max(self.cen - self.com())

   def rg_xy(self):
      d = self.cen[:, :2] - self.com()[:2]
      rg = np.sqrt(np.sum(d**2) / len(d))
      return rg

   def rg_z(self):
      d = self.cen[:, 2] - self.com()[2]
      rg = np.sqrt(np.sum(d**2) / len(d))
      return rg

   def radius_xy_max(self):
      return np.max(self.cen[:, :2] - self.com()[:2])

   def move_by(self, x):
      self.pos = x @ self.pos
      return self

   def move_to(self, x):
      self.pos = x.copy()
      return self

   def move_to_center(self):
      self.pos[:3, 3] = 0
      return self

   def long_axis(self):
      return self.pos @ self.pcavecs[0]

   def long_axis_z_angle(self):
      return np.arccos(abs(self.long_axis()[2])) * 180 / np.pi

   def slide_to(self, other, dirn, radius=_CLASH_RADIUS):
      dirn = np.array(dirn, dtype=np.float64)
      dirn /= np.linalg.norm(dirn)
      delta = bvh.bvh_slide(self.bvh_bb, other.bvh_bb, self.pos, other.pos, radius, dirn)
      if delta < 9e8:
         self.pos[:3, 3] += delta * dirn
      return delta

   def intersect_range(
         self,
         other,
         mindis=2 * _CLASH_RADIUS,
         max_trim=100,
         self_pos=None,
         other_pos=None,
   ):
      self_pos = self.pos if self_pos is None else self_pos
      other_pos = other.pos if other_pos is None else other_pos
      return bvh.isect_range(self.bvh_bb, other.bvh_bb, self_pos, other_pos, mindis,
                             max_trim)

   def intersect(self, other, mindis=2 * _CLASH_RADIUS, self_pos=None, other_pos=None):
      self_pos = self.pos if self_pos is None else self_pos
      other_pos = other.pos if other_pos is None else other_pos
      return bvh.bvh_isect_vec(self.bvh_bb, other.bvh_bb, self_pos, other_pos, mindis)

   def clash_ok(self, *args, **kw):
      return np.logical_not(self.intersect(*args, **kw))

   def distance_to(self, other):
      return bvh.bvh_min_dist(self.bvh_bb, other.bvh_bb, self.pos, other.pos)

   def positioned_coord(self, asym=False):
      n = len(self.coord) // self.nfold if asym else len(self.coord)
      return (self.pos @ self.coord[:n, :, :, None]).squeeze()

   def positioned_cen(self, asym=False):
      n = len(self.stub) // self.nfold if asym else len(self.stub)
      cen = self.stub[:n, :, 3]
      return (self.pos @ cen[..., None]).squeeze()

   def contact_pairs(self, other, maxdis, buf=None):
      if not buf:
         buf = self.pair_buf
      p, o = bvh.bvh_collect_pairs(self.bvh_cen, other.bvh_cen, self.pos, other.pos, maxdis,
                                   buf)
      assert not o
      return p

   def contact_count(self, other, maxdis):
      return bvh.bvh_count_pairs(self.bvh_cen, other.bvh_cen, self.pos, other.pos, maxdis)

   def dump_pdb(self, fname, asym=False):
      from sicdock.io.io_body import dump_pdb_from_bodies

      dump_pdb_from_bodies(fname, [self], symframes(self.sym))

      #      from sicdock.io import pdb_format_atom
      #
      #      s = ""
      #      ia = 0
      #      crd = self.positioned_coord(asym=asym)
      #      cen = self.positioned_cen(asym=asym)
      #      for i in range(len(crd)):
      #         c = self.chain[i]
      #         j = self.resno[i]
      #         aa = self.seq[i]
      #         s += pdb_format_atom(ia=ia + 0, ir=j, rn=aa, xyz=crd[i, 0], c=c, an="N")
      #         s += pdb_format_atom(ia=ia + 1, ir=j, rn=aa, xyz=crd[i, 1], c=c, an="CA")
      #         s += pdb_format_atom(ia=ia + 2, ir=j, rn=aa, xyz=crd[i, 2], c=c, an="C")
      #         s += pdb_format_atom(ia=ia + 3, ir=j, rn=aa, xyz=crd[i, 3], c=c, an="O")
      #         s += pdb_format_atom(ia=ia + 4, ir=j, rn=aa, xyz=crd[i, 4], c=c, an="CB")
      #         s += pdb_format_atom(ia=ia + 5, ir=j, rn=aa, xyz=cen[i], c=c, an="CEN")
      #         ia += 6
      #
      #      with open(fname, "w") as out:
      #         out.write(s)

   def copy(self):
      b = copy.copy(self)
      b.pos = np.eye(4, dtype="f4")  # mutable state can't be same ref as orig
      assert b.pos is not self.pos
      assert b.coord is self.coord
      return b
