import os, _pickle, numpy as np
from pyrosetta import pose_from_file, rosetta, init, version
from pyrosetta.rosetta import core, protocols
from pyrosetta.rosetta.core.scoring.dssp import Dssp
from rpxdock.util import hash_str_to_int

init("-beta -mute all")

def assign_secstruct(pose):
   Dssp(pose).insert_ss_into_pose(pose)

def rosetta_stub_from_numpy_stub(npstub):
   rosstub = ros.core.kinematics.Stub()
   rosstub.M.xx = npstub[0, 0]
   rosstub.M.xy = npstub[0, 1]
   rosstub.M.xz = npstub[0, 2]
   rosstub.M.yx = npstub[1, 0]
   rosstub.M.yy = npstub[1, 1]
   rosstub.M.yz = npstub[1, 2]
   rosstub.M.zx = npstub[2, 0]
   rosstub.M.zy = npstub[2, 1]
   rosstub.M.zz = npstub[2, 2]
   rosstub.v.x = npstub[0, 3]
   rosstub.v.y = npstub[1, 3]
   rosstub.v.z = npstub[2, 3]
   return rosstub

def get_pose_cached(fname, pdbdir="."):
   path = os.path.join(pdbdir, fname)
   h = str(hash_str_to_int(version().encode()))
   ppath = path + "_v" + h + ".pickle"
   if not os.path.exists(ppath):
      pose = pose_from_file(path)
      assign_secstruct(pose)
      with open(ppath, "wb") as out:
         _pickle.dump(pose, out)
         return pose
   with open(ppath, "rb") as inp:
      return _pickle.load(inp)

def get_centroids(pose0, which_resi=None):
   pose = pose0.clone()
   pyrosetta.rosetta.core.util.switch_to_residue_type_set(pose, "centroid")
   if which_resi is None:
      which_resi = list(range(1, pose.size() + 1))
   coords = []
   for ir in which_resi:
      r = pose.residue(ir)
      if not r.is_protein():
         raise ValueError("non-protein residue %s at position %i" % (r.name(), ir))
      cen = r.xyz("CEN")
      coords.append([cen.x, cen.y, cen.z, 1])
   return np.stack(coords).astype("f8")
