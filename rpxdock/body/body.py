import os, copy, numpy as np, rpxdock, logging
from rpxdock import bvh
from rpxdock.util.numeric import pca_eig
from rpxdock import motif
from rpxdock.rosetta import get_bb_coords, get_sc_coords

log = logging.getLogger(__name__)

_CLASH_RADIUS = 1.75

class Body:
   def __init__(self, pdb_or_pose, sym="C1", symaxis=[0, 0, 1], **kw):
      arg = rpxdock.Bunch(kw)

      # pose stuff
      pose = pdb_or_pose
      if isinstance(pdb_or_pose, str):
         import rpxdock.rosetta.triggers_init as ros
         self.pdbfile = pdb_or_pose
         if arg.posecache:
            pose = ros.get_pose_cached(pdb_or_pose)
         else:
            pose = ros.pose_from_file(pdb_or_pose)
            ros.assign_secstruct(pose)
      self.pdbfile = pose.pdb_info().name() if pose.pdb_info() else None
      self.orig_anames, self.orig_coords = get_sc_coords(pose)
      self.seq = np.array(list(pose.sequence()))
      self.ss = np.array(list(pose.secstruct()))
      self.coord = get_bb_coords(pose)
      self.set_asym_body(pose, sym, **kw)

      self.label = arg.label
      if self.label is None and self.pdbfile:
         self.label = os.path.basename(self.pdbfile.rstrip('.gz').rstrip('.pdb'))
      if self.label is None: self.label = 'unk'
      self.components = arg.components if arg.components else []
      self.score_only_ss = arg.score_only_ss if arg.score_only_ss else "EHL"
      self.ssid = motif.ss_to_ssid(self.ss)
      self.chain = np.repeat(0, self.seq.shape[0])
      self.resno = np.arange(len(self.seq))
      self.trim_direction = arg.trim_direction if arg.trim_direction else 'NC'

      self.init_coords(sym, symaxis)

   def init_coords(self, sym, symaxis):
      if isinstance(sym, (int, np.int32, np.int64, np.uint32, np.uint64)):
         sym = "C%i" % sym
      self.sym = sym
      self.symaxis = symaxis
      self.nfold = int(sym[1:])
      if sym and sym[0] == "C" and int(sym[1:]):
         n = self.coord.shape[0]
         nfold = int(sym[1:])
         self.seq = np.array(np.tile(self.seq, nfold))
         self.ss = np.array(np.tile(self.ss, nfold))
         self.ssid = motif.ss_to_ssid(self.ss)
         self.chain = np.repeat(range(nfold), n)
         self.resno = np.tile(range(n), nfold)
         newcoord = np.empty((nfold * n, ) + self.coord.shape[1:])
         newcoord[:n] = self.coord
         for i in range(1, nfold):
            self.pos = rpxdock.homog.hrot(self.symaxis, 360.0 * i / nfold)
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
         if ss in self.score_only_ss:
            which_cen |= self.ss == ss
      which_cen &= ~np.isin(self.seq, ["G", "C", "P"])
      self.which_cen = which_cen
      self.bvh_cen = bvh.bvh_create(self.allcen[:, :3], which_cen)
      self.cen = self.allcen[which_cen]
      self.pos = np.eye(4, dtype="f4")
      self.pcavals, self.pcavecs = pca_eig(self.cen)

   def set_asym_body(self, pose, sym, **kw):
      if isinstance(sym, int): sym = "C%i" % sym
      self.asym_body = self
      if sym != "C1":
         if pose is None:
            log.warning(f'asym_body not built, no pose available')
            self.asym_body = None
         else:
            self.asym_body = Body(pose, "C1", **kw)

   def __len__(self):
      return len(self.seq)

   def strip_data(self):
      self.seq = None
      self.ss = None
      self.ssid = None
      self.coord = None
      self.chain = None
      self.resno = None
      self.allcen = None
      self.cen = None
      self.stub = None
      self.bvh_bb = None
      self.bvh_cen = None
      self.asym_body = None

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

   def intersect_range(self, other, mindis=2 * _CLASH_RADIUS, max_trim=100, self_pos=None,
                       other_pos=None):
      self_pos = self.pos if self_pos is None else self_pos
      other_pos = other.pos if other_pos is None else other_pos
      ntrim = max_trim if 'N' in self.trim_direction else -1
      ctrim = max_trim if 'C' in self.trim_direction else -1
      return bvh.isect_range(self.bvh_bb, other.bvh_bb, self_pos, other_pos, mindis, max_trim,
                             ntrim, ctrim)

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

   def positioned_orig_coords(self):
      return [(self.pos @ x[..., None]).squeeze() for x in self.orig_coords]

   def contact_pairs(self, other, maxdis, buf=None):
      if not buf:
         buf = np.empty((10000, 2), dtype="i4")
      p, o = bvh.bvh_collect_pairs(self.bvh_cen, other.bvh_cen, self.pos, other.pos, maxdis, buf)
      assert not o
      return p

   def contact_count(self, other, maxdis):
      return bvh.bvh_count_pairs(self.bvh_cen, other.bvh_cen, self.pos, other.pos, maxdis)

   def dump_pdb(self, fname, asym=False, **kw):
      # import needs to be here to avoid cyclic import
      from rpxdock.io.io_body import dump_pdb_from_bodies
      bod = [self.asym_body if asym else self]
      return dump_pdb_from_bodies(fname, bod, **kw)

   def copy(self):
      b = copy.copy(self)
      b.pos = np.eye(4, dtype="f4")  # mutable state can't be same ref as orig
      assert b.pos is not self.pos
      assert b.coord is self.coord
      return b

   def copy_with_sym(self, sym, symaxis):
      b = copy.deepcopy(self.asym_body)
      b.pos = np.eye(4, dtype='f4')
      b.init_coords(sym, symaxis)
      return b

   def filter_pairs(self, pairs, score_only_sspair, other=None, sanity_check=True):
      if not other: other = self
      if not score_only_sspair: return pairs
      ss0 = self.ss[pairs[:, 0]]
      ss1 = other.ss[pairs[:, 1]]
      ok = np.ones(len(pairs), dtype=np.bool)
      for sspair in score_only_sspair:
         ss0in0 = np.isin(ss0, sspair[0])
         ss0in1 = np.isin(ss0, sspair[1])
         ss1in0 = np.isin(ss1, sspair[0])
         ss1in1 = np.isin(ss1, sspair[1])
         ok0 = np.logical_and(ss0in0, ss1in1)
         ok1 = np.logical_and(ss1in0, ss0in1)
         ok &= np.logical_or(ok0, ok1)
      if sanity_check:
         sspair = [str(x) + str(y) for x, y in zip(ss0[ok], ss1[ok])]
         for s in set(sspair):
            assert s in score_only_sspair or (s[1] + s[0]) in score_only_sspair
      log.debug(f'filter_pairs {len(pairs)} to {np.sum(ok)}')
      return pairs[ok]
