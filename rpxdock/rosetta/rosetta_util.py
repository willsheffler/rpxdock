import numpy as np, rpxdock as rp
from pyrosetta.rosetta.numeric import xyzVector_double_t as rVec
from pyrosetta.rosetta.numeric import xyzMatrix_double_t as rMat
from pyrosetta import AtomID

def rosetta_init(opts="-beta -mute all"):
   from pyrosetta import init
   init(opts)

def numpy_stub_from_rosetta_stub(rosstub):
   npstub = np.zeros((4, 4))
   for i in range(3):
      npstub[..., i, 3] = rosstub.v[i]
      for j in range(3):
         npstub[..., i, j] = rosstub.M(i + 1, j + 1)
   npstub[..., 3, 3] = 1.0
   return npstub

def get_bb_coords(pose, which_resi=None, recenter_input=False, **kw):
   if which_resi is None:
      which_resi = list(range(1, pose.size() + 1))
   coords = []
   for ir in which_resi:
      r = pose.residue(ir)
      if not r.is_protein():
         raise ValueError("non-protein residue %s at position %i" % (r.name(), ir))
      n, ca, c, o = r.xyz("N"), r.xyz("CA"), r.xyz("C"), r.xyz("O")
      cb = r.xyz("CB") if r.has("CB") else r.xyz("CA")
      coords.append(
         np.array([
            [n.x, n.y, n.z, 1],
            [ca.x, ca.y, ca.z, 1],
            [c.x, c.y, c.z, 1],
            [o.x, o.y, o.z, 1],
            [cb.x, cb.y, cb.z, 1],
         ]))
   coords = np.stack(coords).astype("f8")
   if recenter_input:
      coords = coords.reshape(-1, 4)
      coords[:, :3] -= np.mean(coords[:, :3], axis=0)
      coords = coords.reshape(-1, 5, 4)
   return coords

def get_cb_coords(pose, which_resi=None, recenter_input=False, **kw):
   kw = rp.Bunch(kw)
   if which_resi is None:
      which_resi = list(range(1, pose.size() + 1))
   cbs = []
   for ir in which_resi:
      r = pose.residue(ir)
      if not r.is_protein():
         raise ValueError("non-protein residue %s at position %i" % (r.name(), ir))
      if r.has("CB"):
         cb = r.xyz("CB")
      else:
         cb = r.xyz("CA")
      cbs.append(np.array([cb.x, cb.y, cb.z, 1]))
   coords = np.stack(cbs).astype("f8")
   if recenter_input:
      bb = get_bb_coords(pose, which_resi, **kw.sub(recenter_input=False))
      cen = np.mean(bb.reshape(-1, 4)[:, :3], 0)
      coords[:, :3] -= cen
   return coords

def get_sc_coords(pose, which_resi=None, recenter_input=False, **kw):
   kw = rp.Bunch(kw)
   if which_resi is None:
      which_resi = list(range(1, pose.size() + 1))
   resaname, resacrd = list(), list()
   for ir in which_resi:
      r = pose.residue(ir)
      if not r.is_protein():
         raise ValueError("non-protein residue %s at position %i" % (r.name(), ir))
      anames, crd = list(), list()
      for ia in range(r.natoms()):
         anames.append(r.atom_name(ia + 1))
         xyz = r.xyz(ia + 1)
         crd.append([xyz.x, xyz.y, xyz.z])
      resaname.append(anames)
      hcrd = np.ones((len(anames), 4), dtype='f4')
      hcrd[:, :3] = np.array(crd)
      resacrd.append(hcrd)
   if recenter_input:
      bb = get_bb_coords(pose, which_resi, **kw.sub(recenter_input=False))
      cen = np.mean(bb.reshape(-1, 4)[:, :3], 0)
      for xyz in resacrd:
         xyz[:, :3] -= cen
   return resaname, resacrd

def xform_pose(pose, xform, lb=1, ub=None):
   assert xform.shape == (4, 4)
   if ub is None: ub = pose.size() + 1
   for ir in range(lb, ub):
      res = pose.residue(ir)
      for ia in range(1, res.natoms() + 1):
         old = res.xyz(ia)
         old = np.array([old[0], old[1], old[2], 1])
         new = xform @ old
         pose.set_xyz(AtomID(ia, ir), rVec(new[0], new[1], new[2]))
         # print(ir, ia, new, old)
