import numpy as np, rpxdock as rp
from rpxdock.rotamer.rosetta_rots import (get_rosetta_helix_rots, core, Pose, ala_to_virtCB)

def test_rosetta_rots(dump_pdbs=False):

   pose = Pose()
   core.pose.make_pose_from_sequence(pose, 'AAAAAAAAAAAAAAA', 'fa_standard', auto_termini=False)
   ala_to_virtCB(pose)

   for i in range(1, pose.size() + 1):
      pose.set_phi(i, -47)
      pose.set_psi(i, -57)
      pose.set_omega(i, 180)
   # pose.dump_pdb("refhelix.pdb").
   fields = [
      'coords', 'rotnum', 'atomnum', 'resname', 'atomname', 'atomtype', 'rosetta_atom_type_index'
   ]

   rotdata = get_rosetta_helix_rots(pose, 8)
   for f in fields:
      assert len(rotdata[f]) == 101
   assert 8 == len(rotdata.onebody)

   print(rotdata.onebody)
   assert 0

   rotdata = get_rosetta_helix_rots(pose, 8, extra_rots=[1, 0, 0, 0])
   for f in fields:
      assert len(rotdata[f]) == 290
   assert 23 == len(rotdata.onebody)

   rotdata = get_rosetta_helix_rots(pose, 8, extra_rots=[0, 1, 0, 0])
   for f in fields:
      assert len(rotdata[f]) == 283
   assert 22 == len(rotdata.onebody)

   rotdata = get_rosetta_helix_rots(pose, 8, extra_rots=[1, 1, 0, 0])
   for f in fields:
      assert len(rotdata[f]) == 810
   assert 63 == len(rotdata.onebody)
   print('test_rosetta_rots() DONE')

if __name__ == '__main__':
   test_rosetta_rots(dump_pdbs=True)
   print('test_rosetta_rots.py DONE')