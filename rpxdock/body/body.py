import os, copy, numpy as np, rpxdock, logging, rpxdock as rp
from pandas.core.internals.concat import trim_join_unit
from rpxdock.filter.sscount import secondary_structure_map
from pyrosetta import rosetta as ros

log = logging.getLogger(__name__)
_CLASHRAD = 1.75

#TODO add masking somewhere in this script by take them out of pose when checking for scoring WHS YH

class Body:
   def __init__(
      self,
      source,
      sym="C1",
      symaxis=[0, 0, 1],
      allowed_res=None,
      trim_direction='NC',
      is_subbody=False,
      **kw,
   ):
      kw = rpxdock.Bunch(kw)
      # pose stuff
      pose = source
      if isinstance(source, str):
         import rpxdock.rosetta.triggers_init as ros
         self.pdbfile = source
         if kw.posecache:
            pose = ros.get_pose_cached(source)
         else:
            pose = ros.pose_from_file(source)
            ros.assign_secstruct(pose)
      self.pdbfile = pose.pdb_info().name() if pose.pdb_info() else None
      self.orig_anames, self.orig_coords = rp.rosetta.get_sc_coords(pose, **kw)
      self.seq = np.array(list(pose.sequence()))
      self.ss = np.array(list(pose.secstruct()))
      self.coord = rp.rosetta.get_bb_coords(pose, **kw)
      self.set_asym_body(pose, sym, **kw)

      self.label = kw.label
      if self.label is None and self.pdbfile:
         self.label = os.path.basename(self.pdbfile.replace('.gz', '').replace('.pdb', ''))
      if self.label is None: self.label = 'unk'
      self.components = kw.components if kw.components else []
      self.score_only_ss = kw.score_only_ss if kw.score_only_ss else "EHL"
      self.ssid = rp.motif.ss_to_ssid(self.ss)
      self.chain = np.repeat(0, self.seq.shape[0])
      self.resno = np.arange(len(self.seq))
      # self.trim_direction = trim_direction
      self.trim_direction = kw.trim_direction if kw.trim_direction else 'NC'
      if allowed_res is None:
         self.allowed_residues = np.ones(len(self.seq), dtype='?')
      else:
         self.allowed_residues = np.zeros(len(self.seq), dtype='?')
         for i in allowed_res(self, **kw):
            self.allowed_residues[i - 1] = True
      self.init_coords(sym, symaxis, **kw)

      self.is_subbody = is_subbody
      if not is_subbody:
         self.trimN_subbodies, self.trimC_subbodies = get_trimming_subbodies(self, pose, **kw)
         print('trimN_subbodies', len(self.trimN_subbodies))
         print('trimC_subbodies', len(self.trimC_subbodies))
         # assert 0

   def init_coords(self, sym, symaxis, xform=np.eye(4), ignored_aas='CGP', **kw):
      kw = rp.Bunch(kw)
      if isinstance(sym, np.ndarray):
         assert len(sym) == 1
         sym = sym[0]
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
         self.ssid = rp.motif.ss_to_ssid(self.ss)
         self.chain = np.repeat(range(nfold), n)
         self.resno = np.tile(range(n), nfold)
         newcoord = np.empty((nfold * n, ) + self.coord.shape[1:])
         newcoord[:n] = self.coord
         new_orig_coords = self.orig_coords
         for i in range(1, nfold):
            self.pos = rpxdock.homog.hrot(self.symaxis, 360.0 * i / nfold)
            newcoord[i * n:][:n] = self.positioned_coord()
            new_orig_coords.extend(self.positioned_orig_coords())
         self.coord = (xform @ newcoord[:, :, :, None]).reshape(-1, 5, 4)
         self.orig_coords = [(xform @ oc[:, :, None]).reshape(-1, 4) for oc in new_orig_coords]
      else:
         raise ValueError("unknown symmetry: " + sym)
      assert len(self.seq) == len(self.coord)
      assert len(self.ss) == len(self.coord)
      assert len(self.chain) == len(self.coord)

      self.nres = len(self.coord)
      self.stub = rp.motif.bb_stubs(self.coord)
      ids = np.repeat(np.arange(self.nres, dtype=np.int32), self.coord.shape[1])
      self.bvh_bb = rp.BVH(self.coord[..., :3].reshape(-1, 3), [], ids)
      self.bvh_bb_atomno = rp.BVH(self.coord[..., :3].reshape(-1, 3), [])
      self.allcen = self.stub[:, :, 3]
      which_cen = np.repeat(False, len(self.ss))
      for ss in "EHL":
         if ss in self.score_only_ss:
            which_cen |= self.ss == ss
      which_cen &= ~np.isin(self.seq, [list(ignored_aas)])

      if not hasattr(self, 'allowed_residues'):
         self.allowed_residues = np.ones(len(self.seq), dtype='?')
      allowed_res = self.allowed_residues
      nallow = len(self.allowed_residues)
      if nallow > len(self.ss):
         allowed_res = self.allowed_residues[:len(self.ss)]
      elif nallow < len(self.ss):
         allowed_res = np.tile(self.allowed_residues, len(self.ss) // nallow)
      self.which_cen = which_cen & allowed_res

      self.bvh_cen = rp.BVH(self.allcen[:, :3], self.which_cen)
      self.cen = self.allcen[self.which_cen]
      self.pos = np.eye(4, dtype="f4")
      self.pcavals, self.pcavecs = rp.util.numeric.pca_eig(self.cen)

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
      return self.bvh_bb.com()

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
      self.pos[:, 3] = -self.bvh_bb.com()
      return self

   def long_axis(self):
      return self.pos @ self.pcavecs[0]

   def long_axis_z_angle(self):
      return np.arccos(abs(self.long_axis()[2])) * 180 / np.pi

   def slide_to(self, other, dirn, radius=_CLASHRAD):
      dirn = np.array(dirn, dtype=np.float64)
      dirn /= np.linalg.norm(dirn)
      delta = rp.bvh.bvh_slide(self.bvh_bb, other.bvh_bb, self.pos, other.pos, radius, dirn)
      if delta < 9e8:
         self.pos[:3, 3] += delta * dirn
      return delta

   def intersect_range(self, other, xself=None, xother=None, mindis=2 * _CLASHRAD, max_trim=100,
                       nasym1=None, debug=False, **kw):
      '''
      :param other:
      :param xself: body1 pos
      :param xother: body2 pos
      :param mindis: clash distance
      :param max_trim: max n of residues to trim (similar to ntrim and ctrim)
      :param nasym1: n res in asu
      :param debug:
      :param kw:
      :return: takes bvh (bounding volume hierarchies) written in c++, and subdivides spheres to check for intersection
      among smaller and smaller spheres (basically evaluates intersection between clouds of points)
      '''
      if nasym1 is None: nasym1 = self.asym_body.nres
      xself = self.pos if xself is None else xself
      xother = other.pos if xother is None else xother
      ntrim = max_trim if 'N' in self.trim_direction else -1
      ctrim = max_trim if 'C' in self.trim_direction else -1
      # print('intersect_range', mindis, nasym1, self.bvh_bb.max_id(), max_trim, ntrim, ctrim)
      trim = rp.bvh.isect_range(self.bvh_bb, other.bvh_bb, xself, xother, mindis, max_trim, ntrim,
                                ctrim, nasym1=nasym1)

      if debug:
         ok = np.logical_and(trim[0] >= 0, trim[1] >= 0)
         xotherok = xother[ok] if xother.ndim == 3 else xother
         xselfok = xself[ok] if xself.ndim == 3 else xself
         # print(xselfok.shape, trim[0].shape, trim[0][ok].shape)
         # print(xotherok.shape, trim[1].shape, trim[1][ok].shape)
         clash, ids = rp.bvh.bvh_isect_fixed_range_vec(self.bvh_bb, other.bvh_bb, xselfok,
                                                       xotherok, mindis, trim[0][ok], trim[1][ok])
         # print(np.sum(clash) / len(clash))
         assert not np.any(clash)

      return trim

   def intersect(self, other, xself=None, xother=None, mindis=2 * _CLASHRAD, **kw):
      xself = self.pos if xself is None else xself
      xother = other.pos if xother is None else xother
      return rp.bvh.bvh_isect_vec(self.bvh_bb, other.bvh_bb, xself, xother, mindis)

   def clash_ok(self, *args, **kw):
      return np.logical_not(self.intersect(*args, **kw))

   def distance_to(self, other):
      return rp.bvh.bvh_min_dist(self.bvh_bb, other.bvh_bb, self.pos, other.pos)

   def positioned_coord(self, asym=False, pos=None):
      pos = self.pos if pos is None else pos
      assert pos.shape == (4, 4)
      n = len(self.coord) // self.nfold if asym else len(self.coord)
      return (self.pos @ self.coord[:n, :, :, None]).squeeze()

      foo = self.coord[:n, :, :]
      # print(pos.shape, foo.shape)
      bar = pos @ foo
      # print(bar.shape)
      # assert 0
      return bar.squeeze()

   def positioned_coord_atomno(self, i):
      return self.pos @ self.coord.reshape(-1, 4)[i]

   def positioned_cen(self, asym=False):
      n = len(self.stub) // self.nfold if asym else len(self.stub)
      cen = self.stub[:n, :, 3]
      return (self.pos @ cen[..., None]).squeeze()

   def positioned_orig_coords(self):
      return [(self.pos @ x[..., None]).squeeze() for x in self.orig_coords]

   def contact_pairs(self, other, maxdis, buf=None, use_bb=False, atomno=False):
      if not buf:
         buf = np.empty((10000, 2), dtype="i4")
      pairs, overflow = rp.bvh.bvh_collect_pairs(
         self.bvh_bb_atomno if atomno else (self.bvh_bb if use_bb else self.bvh_cen),
         other.bvh_bb_atomno if atomno else (other.bvh_bb if use_bb else other.bvh_cen),
         self.pos,
         other.pos,
         maxdis,
         buf,
      )
      assert not overflow
      return pairs

   def contact_count(self, other, maxdis):
      return rp.bvh.bvh_count_pairs(self.bvh_cen, other.bvh_cen, self.pos, other.pos, maxdis)

   def dump_pdb(self, fname, **kw):
      # import needs to be here to avoid cyclic import
      from rpxdock.io.io_body import dump_pdb_from_bodies
      return dump_pdb_from_bodies(fname, [self], **kw)

   def str_pdb(self, **kw):
      # import needs to be here to avoid cyclic import
      from rpxdock.io.io_body import dump_pdb_from_bodies
      return rp.io.make_pdb_from_bodies([self], **kw)

   def copy(self):
      b = copy.copy(self)
      b.pos = np.eye(4, dtype="f4")  # mutable state can't be same ref as orig
      assert b.pos is not self.pos
      assert b.coord is self.coord
      return b

   def copy_with_sym(self, sym, symaxis=[0, 0, 1]):
      b = copy.deepcopy(self.asym_body)
      b.pos = np.eye(4, dtype='f4')
      b.init_coords(sym, symaxis)
      b.asym_body = self.asym_body
      return b

   def copy_xformed(self, xform):
      b = copy.deepcopy(self.asym_body)
      b.pos = np.eye(4, dtype='f4')
      b.init_coords('C1', [0, 0, 1], xform)
      b.asym_body = b
      return b

   def filter_pairs(self, pairs, score_only_sspair, other=None, lbub=None, sanity_check=True):
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
      if lbub:
         assert 0
      else:
         return pairs[ok]

   def __repr__(self):
      source = self.pdbfile if self.pdbfile else '<rosetta Pose of unknown origin>'
      return f'Body(source="{source}")'

def get_trimming_subbodies(body, pose, debug=False, **kw):
   kw = rp.Bunch(kw)
   if kw.helix_trim_max == 0 or kw.helix_trim_max is None:
      return [], []
   print('body.ss', ''.join(body.ss))
   ssmap = secondary_structure_map()
   ssmap.map_body_ss(body)
   # print(ssmap.ss_index)
   # print(ssmap.ss_type_assignments)
   # print(ssmap.ss_element_start)
   # print(ssmap.ss_element_end)
   ssid = np.array(ssmap.ss_index)
   sstype = np.array(ssmap.ss_type_assignments)
   selection = sstype == 'H'
   ssid = ssid[selection]
   sstype = sstype[selection]
   lb = np.array(ssmap.ss_element_start)[selection]
   ub = np.array(ssmap.ss_element_end)[selection] + 1
   # TODO direction matters here !!!
   print(lb)
   print(ub)
   # print()

   # trim_place_nc = 197  # len(body) - max_trim
   # trim_place_cn = 100  # max_trim
   lb_nc = lb.copy()
   ub_nc = ub.copy()
   lb_nc[0] = 0
   lb_nc[1:] = ub[:-1]

   #  # # print(ssid)
   # print('lb_nc', lb_nc)
   # print('ub_nc', ub_nc)
   # nhelix = np.sum(lb_nc >= trim_place_nc)
   # print(nhelix)
   # print(lb_nc[-nhelix:])
   # print(ub_nc[-nhelix:])
   # print()
   # assert nhelix == 6

   lb_cn = lb.copy()
   ub_cn = ub.copy()
   # print(len(body.ss))
   ub_cn[-1] = len(body.ss)
   ub_cn[:-1] = lb_cn[1:]
   # # lb_cn[0] = 0
   # # lb_cn[1:] = ub[:-1]
   # # print(ssid)
   # print('lb_cn', lb_cn)
   # print('ub_cn', ub_cn)
   # nhelix = np.sum(ub_cn <= trim_place_cn)
   # print(nhelix)
   # print(lb_cn[:nhelix])
   # print(ub_cn[:nhelix])
   # print()
   # print()
   # assert nhelix == 5

   hmt = kw.helix_trim_max
   htnie = kw.helix_trim_nres_ignore_end

   trimC_subbodies = list()
   p = ros.core.pose.Pose()
   ros.core.pose.append_subpose_to_pose(p, pose, 1, ub_nc[-hmt - 1])
   trimC_subbodies.append(Body(p, is_subbody=True))
   for i, (start, end) in enumerate(zip(lb_nc[-hmt:], ub_nc[-hmt:])):
      p = ros.core.pose.Pose()
      ros.core.pose.append_subpose_to_pose(p, pose, start + 1, end - htnie)
      # print('nc_%i.pdb' % i)
      trimC_subbodies.append(Body(p, is_subbody=True))
   trimN_subbodies = list()
   p = ros.core.pose.Pose()
   ros.core.pose.append_subpose_to_pose(p, pose, lb_cn[hmt] + htnie, pose.size())
   trimN_subbodies.append(Body(p, is_subbody=True))
   for i, (start, end) in enumerate(zip(lb_cn[:hmt], ub_cn[:hmt])):
      p = ros.core.pose.Pose()
      ros.core.pose.append_subpose_to_pose(p, pose, start + 1 + htnie, end)
      # print('cn_%i.pdb' % i)
      trimN_subbodies.append(Body(p, is_subbody=True))

   if debug:
      for i, b in enumerate(trimC_subbodies):
         print('dump body %i' % i)
         b.dump_pdb('trimC_%i.pdb' % i)

      for i, b in enumerate(trimN_subbodies):
         print('dump body %i' % i)
         b.dump_pdb('trimN_%i.pdb' % i)

   return trimN_subbodies, trimC_subbodies