import os, _pickle, numpy as np, rpxdock as rp
from pyrosetta import Pose, pose_from_file, rosetta, init, version, AtomID
from pyrosetta.rosetta import core, protocols, numeric
from pyrosetta.rosetta.core.scoring.dssp import Dssp
from pyrosetta.rosetta.core.scoring import get_score_function

def get_init_string():
   s = ' -beta '
   s += ' -mute all '
   s += ' -extra_res_fa '
   s += ' '.join(rp.data.rosetta_params_files())
   # s += ' -extra_patch_fa '
   # s += ' '.join(rp.data.rosetta_patch_files())
   # s += ' -include_patches VIRTUAL_CB'
   print('===================== rosetta flags ==========================')
   print(s)
   print('==============================================================')
   return s

init(get_init_string())
default_sfxn = get_score_function()
chem_manager = core.chemical.ChemicalManager.get_instance()
rts_fastd = chem_manager.residue_type_set('fa_standard')

def assign_secstruct(pose):
   Dssp(pose).insert_ss_into_pose(pose)

def rosetta_stub_from_numpy_stub(npstub):
   rosstub = core.kinematics.Stub()
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

def create_residue(resname, typeset='fa_standard'):
   chem_manager = rosetta.core.chemical.ChemicalManager
   rts = chem_manager.get_instance().residue_type_set(typeset)
   # print(rts)
   rfactory = rosetta.core.conformation.ResidueFactory
   return rfactory.create_residue(rts.name_map(resname))

def get_pose_cached(fname, pdbdir="."):
   path = os.path.join(pdbdir, fname)
   h = str(rp.util.hash_str_to_int(version().encode()))
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

def remove_terminus_variants(pose):
   for ires in range(1, pose.size() + 1):
      if (pose.residue(ires).has_variant_type(core.chemical.UPPER_TERMINUS_VARIANT)):
         core.pose.remove_variant_type_from_pose_residue(
            pose, core.chemical.UPPER_TERMINUS_VARIANT, ires)
      if (pose.residue(ires).has_variant_type(core.chemical.LOWER_TERMINUS_VARIANT)):
         core.pose.remove_variant_type_from_pose_residue(
            pose, core.chemical.LOWER_TERMINUS_VARIANT, ires)
      if (pose.residue(ires).has_variant_type(core.chemical.CUTPOINT_LOWER)):
         core.pose.remove_variant_type_from_pose_residue(pose, core.chemical.CUTPOINT_LOWER, ires)
      if (pose.residue(ires).has_variant_type(core.chemical.CUTPOINT_UPPER)):
         core.pose.remove_variant_type_from_pose_residue(pose, core.chemical.CUTPOINT_UPPER, ires)
