import os, copy, logging, io, tempfile, sys
import numpy as np
import rpxdock as rp
# from pandas.core.internals.concat import trim_join_unit
from rpxdock.filter.sscount import secondary_structure_map

import willutil as wu
from willutil import Bunch

log = logging.getLogger(__name__)
_CLASHRAD = 4.5

#TODO add masking somewhere in this script by take them out of pose when checking for scoring WHS YH

class Body:
   @wu.timed
   def __init__(
      self,
      source,
      sym="C1",
      symaxis=[0, 0, 1],
      trim_direction='NC',
      is_subbody=False,
      modified_term=[False, False],
      og_seqlen=None,
      dont_use_rosetta=False,
      ignored_aas='CPG',
      **kw,
   ):
      kw = wu.Bunch(kw)
      self.pos = np.eye(4)
      self.is_subbody = is_subbody
      pose = self.set_pose_info(source, sym=sym, userosetta=not dont_use_rosetta, **kw)
      self.modified_term = modified_term
      self.og_seqlen = og_seqlen or pose.size()

      self.label = kw.get('label')
      if self.label is None and self.pdbfile:
         self.label = os.path.basename(self.pdbfile.replace('.gz', '').replace('.pdb', ''))
      if self.label is None: self.label = ''
      # assert 'score_only_ss' in kw
      self.components = kw.get('components', [])
      self.score_only_ss = kw.get('score_only_ss', 'EHL')
      self.trim_direction = kw.get('trim_direction', 'NC')
      self.ignored_aas = ignored_aas
      self.set_allowed_residues(**kw)
      self.init_coords(sym, symaxis, **kw)
      if not is_subbody and not dont_use_rosetta:
         self.trimN_subbodies, self.trimC_subbodies = get_trimming_subbodies(self, pose, **kw)

   @wu.timed
   def set_pose_info(self, source, sym, extract_chain=None, userosetta=True, **kw):
      # pose stuff
      if userosetta:
         try:
            import rpxdock.rosetta.triggers_init as ros
         except ImportError:
            userosetta = False
      ic(type(source))
      kw = Bunch(kw)
      pose = source
      self.rawpdb = None
      self.rawcoords = None
      if isinstance(source, str):
         # import rpxdock.rosetta.triggers_init as ros
         self.pdbfile = source
         if userosetta:
            if kw.get('posecache'):
               pose = ros.get_pose_cached(source)
            assert os.path.exists(source)
            pose = ros.pose_from_file(source)
            ros.assign_secstruct(pose)
         else:
            pose = wu.NotPose(source, **kw)
            self.rawpdb = None if pose.coordsonly else pose.rawpdb
      elif isinstance(source, io.BytesIO):
         import rpxdock.rosetta.triggers_init as ros
         self.pdbfile = kw.source_filename
         with tempfile.TemporaryDirectory() as td:
            with open(td + '/tmp.pdb', 'w') as out:
               out.write(source.read().decode())
            pose = ros.pose_from_file(td + '/tmp.pdb')
            ros.assign_secstruct(pose)
      elif isinstance(source, np.ndarray):
         pose = wu.NotPose(coords=source, **kw)
         self.rawpdb = None
         self.rawcoords = source.copy()
      elif isinstance(source, wu.pdb.pdbfile.PDBFile):
         pose = wu.NotPose(pdb=source, **kw)
         self.rawpdb = source
      elif userosetta:
         import pyrosetta
         if not isinstance(source, pyrosetta.rosetta.core.pose.Pose):
            raise ValueError(f'Body cant understand source {type(source)}')
      elif 'torch' in sys.modules:
         import torch
         if isinstance(source, torch.Tensor):
            source = source.detach().numpy()
            pose = wu.NotPose(coords=source, **kw)
            self.rawpdb = None
            self.rawcoords = source.copy()
      else:
         raise ValueError(f'Body cant understand source {type(source)}')
      wu.checkpoint(kw, 'read_source')

      self.crystinfo = None
      if pose.pdb_info() is not None:
         ci = pose.pdb_info().crystinfo()
         if ci is not None:
            self.crystinfo = wu.Bunch(spacegroup=ci.spacegroup(), cell=[ci.A(), ci.B(), ci.C()],
                                      ang=[ci.alpha(), ci.beta(), ci.gamma()])
      self.pdbfile = pose.pdb_info().name() if pose.pdb_info() else None

      if extract_chain is not None:
         assert extract_chain == 0
         for nres in range(pose.size()):
            if pose.chain(nres + 1) != 1 + extract_chain: break

         if userosetta:
            p = ros.core.pose.Pose()
            ros.core.pose.append_subpose_to_pose(p, pose, 1, nres)
            # ic(p.size(), pose.size())
            # assert p.size() in (pose.size(), pose.size() / 4)
            pose = p
            ros.assign_secstruct(pose)
         if not userosetta:
            pose = pose.extract(chain=extract_chain, **kw)

      # timer.checkpoint('body load file')

      self.orig_anames, self.orig_coords = rp.rosetta.get_sc_coords(pose, **kw)
      self.seq = np.array(list(pose.sequence()))
      self.ss = np.array(list(pose.secstruct()))

      # what is this about? seems bugged? how should rest of body know some is trimmed???
      # self.ss = rp.util.trim_ss(self, **kw)
      assert self.is_subbody or np.sum(self.ss == 'L') < len(self.ss), 'body is all loops and not sub-body!!'
      self.ssid = rp.motif.ss_to_ssid(self.ss)
      self.chain = np.repeat(0, self.seq.shape[0])
      self.resno = np.arange(len(self.seq))
      # if userosetta:
      self.coord = rp.rosetta.get_bb_coords(pose, **kw)
      self.coord = np.ascontiguousarray(self.coord)
      # ic('rosetta', self.coord.shape)
      # else:
      # self.coord = pose.bb()
      self.set_asym_body(pose, sym, **kw)
      return pose

   @wu.timed
   def set_allowed_residues(
      self,
      allowed_res=None,
      required_res_sets=None,
      **kw,
   ):
      # ic(required_res_sets)

      self.allowed_residues = np.zeros(len(self), dtype='?')
      if allowed_res is None:
         if True in self.modified_term:
            for j in range(self.og_seqlen):
               self.allowed_residues[j] = True
         else:
            self.allowed_residues = np.ones(len(self), dtype='?')
      else:
         if callable(allowed_res):
            allowed_res = allowed_res(self, **kw)
         for j in allowed_res:
            self.allowed_residues[j - 1] = True
      #ic(self.allowed_residues.shape, np.sum(self.allowed_residues))

      # list of sets of residues
      # participating protocols should require at least one residue at iface for ALL sets
      # sets represnted at PHMaps because I never implement a set... just ignore value
      # us np.sum(mymap.has(value)) >= num_required to test
      logging.debug(required_res_sets)
      if required_res_sets is None:
         required_res_sets = []
      # for s in required_res_sets:
      # required_res_sets = [np.asarray(x, dtype='i4') for x in required_res_sets]
      self.required_res_sets = list()
      for s in required_res_sets:
         if s is None: continue
         if callable(s): s = s(self, **kw)
         s = np.asarray(list(s), dtype='i4')
         self.allowed_residues[s] = True
         phmap = rp.phmap.PHMap_i4i4()
         phmap[s] = np.repeat(np.array([12345], dtype='i4'), len(s))
         self.required_res_sets.append(phmap)
         # ic(phmap.keys())

      #ic(self.allowed_residues.shape, np.sum(self.allowed_residues))
      #for s in self.required_res_sets:
      #ic(s.keys())

   @wu.timed
   def init_coords(
         self,
         sym,
         symaxis=np.array([0, 0, 1, 0]),
         xform=np.eye(4),
         ignored_aas=None,
         **kw,
   ):
      kw = wu.Bunch(kw)
      if ignored_aas is None:
         ignored_aas = self.ignored_aas if hasattr(self, 'ignored_aas') else 'CGP'

      if isinstance(sym, np.ndarray):
         assert len(sym) == 1
         sym = sym[0]
      if isinstance(sym, (int, np.int32, np.int64, np.uint32, np.uint64)):
         sym = "C%i" % sym
      sym = sym.upper()
      self.sym = sym
      self.symaxis = symaxis
      self.nfold = int(sym[1:])
      self.symframes = np.eye(4).reshape(1, 4, 4)
      self.nterms = [self.coord[0, 0]]
      self.cterms = [self.coord[-1, 2]]
      self._pos = np.eye(4)
      if sym and sym[0] == "C" and int(sym[1:]) > 0:
         n = self.coord.shape[0]
         nfold = int(sym[1:])
         self.seq = np.array(np.tile(self.seq, nfold))
         self.ss = np.array(np.tile(self.ss, nfold))
         self.ssid = rp.motif.ss_to_ssid(self.ss)
         self.chain = np.repeat(range(nfold), n)
         self.resno = np.tile(range(n), nfold)
         newcoord = np.empty((nfold * n, ) + self.coord.shape[1:])
         newcoord[:n] = self.coord
         # ic(self.coord.shape)
         new_orig_coords = self.orig_coords
         symframes = [np.eye(4)]
         for i in range(1, nfold):
            symframes.append(wu.hrot(self.symaxis, 360.0 * i / nfold))
            self.pos = symframes[-1]
            newcoord[i * n:][:n] = self.positioned_coord()
            new_orig_coords.extend(self.positioned_orig_coords())
            self.nterms.append(symframes[-1] @ self.nterms[0])
            self.cterms.append(symframes[-1] @ self.cterms[0])
         self.symframes = np.stack(symframes)
         self.coord = (xform @ newcoord[:, :, :, None]).reshape(-1, 5, 4)
         self.coord = np.ascontiguousarray(self.coord)
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
      self._symcom = wu.homog.hxform(self.symframes, wu.homog.htrans(self.asym_body.bvh_bb.com()))
      self.allcen = self.stub[:, :, 3]
      which_cen = np.repeat(False, len(self))
      for ss in "EHL":
         if ss in self.score_only_ss:
            which_cen |= self.ss == ss
      which_cen &= ~np.isin(self.seq, [list(ignored_aas)])
      if not hasattr(self, 'allowed_residues'):
         self.allowed_residues = np.ones(len(self.seq), dtype='?')
      allowed_res = self.allowed_residues
      nallow = len(self.allowed_residues)
      if nallow > len(self):
         allowed_res = self.allowed_residues[:len(self)]
      elif nallow < len(self):
         allowed_res = np.tile(self.allowed_residues, len(self) // nallow)
      self.which_cen = which_cen & allowed_res
      assert np.any(self.which_cen)
      self.bvh_cen = rp.BVH(self.allcen[:, :3], self.which_cen)
      # ic(len(self.bvh_cen))
      self.cen = self.allcen[self.which_cen]
      self.pos = np.eye(4, dtype="f4")
      self.pcavals, self.pcavecs = rp.util.numeric.pca_eig(self.cen)

   @wu.timed
   def copy_with_sym(self, sym, symaxis=[0, 0, 1], newaxis=None, phase=0):
      x = np.eye(4)
      if newaxis is not None:
         x = wu.homog.align_vector(symaxis, newaxis)
         x = wu.hrot(newaxis, phase) @ x
      b = copy.deepcopy(self.asym_body)

      if isinstance(sym, str): sym = int(sym[1:])
      if isinstance(sym, np.ndarray): sym = int(sym[0])
      if isinstance(sym, (np.int32, np.int64)): sym = int(sym)
      assert isinstance(sym, int)
      if sym == 1: return b
      b.pos = np.eye(4, dtype='f4')
      b.asym_body = self.asym_body
      b.init_coords(sym, symaxis, xform=x)
      return b

   @wu.timed
   def copy_xformed(self, xform, **kw):
      b = copy.deepcopy(self.asym_body)
      b.pos = np.eye(4, dtype='f4')
      b.asym_body = b
      b.init_coords('C1', [0, 0, 1], xform, **kw)
      assert len(self.bvh_cen) == len(b.bvh_cen)
      return b

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

   @property
   def pos(self):
      if not hasattr(self, '_pos'):
         self._pos = np.eye(4)
      return self._pos

   @pos.setter
   def pos(self, xform):
      assert xform.shape == (4, 4)
      self._pos = xform

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
      assert x.shape == (4, 4)
      self.pos = x[:]
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

   def intersect_range(self, other, xself=None, xother=None, mindis=2 * _CLASHRAD, max_trim=100, nasym1=None,
                       debug=False, **kw):
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
      trim = rp.bvh.isect_range(self.bvh_bb, other.bvh_bb, xself, xother, mindis, max_trim, ntrim, ctrim, nasym1=nasym1)

      if debug:
         ok = np.logical_and(trim[0] >= 0, trim[1] >= 0)
         xotherok = xother[ok] if xother.ndim == 3 else xother
         xselfok = xself[ok] if xself.ndim == 3 else xself
         # print(xselfok.shape, trim[0].shape, trim[0][ok].shape)
         # print(xotherok.shape, trim[1].shape, trim[1][ok].shape)
         clash, ids = rp.bvh.bvh_isect_fixed_range_vec(self.bvh_bb, other.bvh_bb, xselfok, xotherok, mindis,
                                                       trim[0][ok], trim[1][ok])
         # print(np.sum(clash) / len(clash))
         assert not np.any(clash)

      return trim

   def intersect(self, other, xself=None, xother=None, mindis=2 * _CLASHRAD, **kw):
      xself = self.pos if xself is None else xself
      xother = other.pos if xother is None else xother
      return rp.bvh.bvh_isect_vec(self.bvh_bb, other.bvh_bb, xself, xother, mindis)

   def clash_ok(self, *args, **kw):
      return np.logical_not(self.intersect(*args, **kw))

   def distance_to(self, other, spos=None, opos=None):
      spos = self.pos if spos is None else spos
      opos = self.pos if opos is None else opos
      dist, i1, i2 = rp.bvh.bvh_min_dist_vec(self.bvh_bb, other.bvh_bb, spos, opos)
      return dist

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

   def contact_pairs(self, other, pos1=None, pos2=None, maxdis=10, buf=None, use_bb=False, atomno=False):
      if pos1 is None: pos1 = self.pos
      if pos2 is None: pos2 = other.pos
      if not buf: buf = np.empty((10000, 2), dtype="i4")
      pairs, overflow = rp.bvh.bvh_collect_pairs(
         self.bvh_bb_atomno if atomno else (self.bvh_bb if use_bb else self.bvh_cen), other.bvh_bb_atomno if atomno else
         (other.bvh_bb if use_bb else other.bvh_cen), pos1, pos2, maxdis, buf)
      assert not overflow
      return pairs

   def contact_count(self, other, pos1=None, pos2=None, maxdis=10, debug=False):
      if pos1 is None:
         if debug: print("pos1 = self.pos")
         pos1 = self.pos
      if pos2 is None:
         if debug: print("pos2 = other.pos")
         pos2 = other.pos

      if debug:
         print("self.bvh_cen")
         print(self.bvh_cen)
         print("other.bvh_cen")
         print(other.bvh_cen)
         print("pos1")
         print(pos1)
         print("pos2")
         print(pos2)
         print("maxdis")
         print(maxdis)

      return rp.bvh.bvh_count_pairs_vec(self.bvh_cen, other.bvh_cen, pos1, pos2, maxdis)

   def contact_count_bb(self, other, pos1=None, pos2=None, maxdis=10):
      if pos1 is None: pos1 = self.pos
      if pos2 is None: pos2 = other.pos
      return rp.bvh.bvh_count_pairs_vec(self.bvh_bb, other.bvh_bb, pos1, pos2, maxdis)

   def dump_pdb(self, fname, **kw):
      # import needs to be here to avoid cyclic import
      from rpxdock.io.io_body import dump_pdb_from_bodies
      return dump_pdb_from_bodies(fname, [self], **kw)

   def str_pdb(self, **kw):
      # import needs to be here to avoid cyclic import
      from rpxdock.io.io_body import dump_pdb_from_bodies
      self.pos = np.eye(4)
      return rp.io.make_pdb_from_bodies([self], **kw)

   def copy(self):
      b = copy.copy(self)
      b.pos = np.eye(4, dtype="f4")  # mutable state can't be same ref as orig
      assert b.pos is not self.pos
      assert b.coord is self.coord
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

   def source(self):
      return self.pdbfile if self.pdbfile else '<rosetta Pose of unknown origin>'

   def __repr__(self):
      return f'Body(source="{self.source()}")'
      source = self.pdbfile if self.pdbfile else '<rosetta Pose of unknown origin>'
      return f'Body(source="{source}")'

   # Make copy of body that had helix appended at termini, such that new body
   # is a copy of original but does not include info about the helical residues
   def copy_exclude_term_res(self):
      newbody = self.copy()
      for attribute in dir(self):
         val = getattr(self, attribute)
         if type(val) == list:
            if len(val) != self.og_seqlen and len(val) == self.nres:
               setattr(newbody, attribute, copy.deepcopy(val)[:(self.og_seqlen)])
         elif isinstance(val, np.ndarray):
            if len(val) != self.og_seqlen and len(val) == self.nres:
               setattr(newbody, attribute, val.copy()[:(self.og_seqlen)])
      newbody.nres = self.og_seqlen
      newbody.modified_term = [False, False]  #no longer includes modified termini
      return newbody

   def symcom(self, pos=np.eye(4).reshape(-1, 4, 4), flat=False):
      # print('symcom', pos.shape)
      return wu.homog.hxform(pos, self._symcom, outerprod=True, flat=flat)

   def symcomdist(self, pos=np.eye(4).reshape(-1, 4, 4), pos2=None, mask=False):
      if pos2 is None: pos2 = pos
      else: mask = False
      coms = self.symcom(pos, flat=False)
      coms2 = self.symcom(pos2, flat=False)
      dist = wu.homog.hdist(coms, coms2)
      if mask and pos2 is not None:
         for i in range(len(pos)):
            for j in range(i + 1):
               dist[i, :, j, :] = 9e9
      return dist

def get_trimming_subbodies(body, pose, debug=False, **kw):
   from pyrosetta import rosetta as ros
   kw = Bunch(kw, _strict=False)
   if kw.helix_trim_max == 0 or kw.helix_trim_max is None:
      return [], []
   # print('body.ss', ''.join(body.ss))
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

def get_body_cached(
      fname,
      csym='c1',
      xaln=np.eye(4),
      cachedir='.rpxcache',
      **kw,
):
   os.makedirs(cachedir, exist_ok=True)
   cache_fname = os.path.join(cachedir, os.path.basename(fname))
   cache_fname += '_' + csym
   cache_fname += '_xaln%i' % rp.util.hash_str_to_int(repr(xaln))
   cache_fname += '.body.pickle'
   if os.path.exists(cache_fname):
      body = rp.load(cache_fname)
   else:
      body = Body(fname, **kw)
      body = body.copy_xformed(xaln)
      body = body.copy_with_sym(csym)
      rp.dump(body, cache_fname)
   return body
