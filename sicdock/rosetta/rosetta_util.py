import numpy as np

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

def get_bb_coords(pose, which_resi=None):
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
   return np.stack(coords).astype("f8")

def get_cb_coords(pose, which_resi=None):
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
   return np.stack(cbs).astype("f8")

def get_sc_coords(pose, which_resi=None):
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
   return resaname, resacrd
