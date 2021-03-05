import numpy as np, rpxdock as rp
from rpxdock.rotamer.rosetta_rots import (get_rosetta_rots, core, Pose, ala_to_virtCB)

def test_rosetta_rots_1res(**kw):

   pose = Pose()
   core.pose.make_pose_from_sequence(pose, 'AAAAAAAAAAAAAAA', 'fa_standard', auto_termini=False)
   ala_to_virtCB(pose)

   for i in range(1, pose.size() + 1):
      pose.set_phi(i, -47)
      pose.set_psi(i, -57)
      pose.set_omega(i, 180)
   sfxn = rp.rosetta.get_score_function()
   sfxn.score(pose)
   # pose.dump_pdb("refhelix.pdb").

   fields = [
      'coords', 'rotnum', 'atomnum', 'resname', 'atomname', 'atomtype', 'rosetta_atom_type_index'
   ]

   t = rp.Timer().start()

   rotdata = get_rosetta_rots(pose, [8], sfxn, **kw)
   for f in fields:
      assert len(rotdata[f]) == 105
   assert 9 == len(rotdata.onebody)
   t.checkpoint('no ex')

   rotdata = get_rosetta_rots(pose, [8], sfxn, extra_rots=[1, 0, 0, 0], **kw)
   for f in fields:
      assert len(rotdata[f]) == 294
   assert 24 == len(rotdata.onebody)
   t.checkpoint('ex1')

   rotdata = get_rosetta_rots(pose, [8], sfxn, extra_rots=[0, 1, 0, 0], **kw)
   for f in fields:
      assert len(rotdata[f]) == 287
   assert 23 == len(rotdata.onebody)
   t.checkpoint('ex2')

   rotdata = get_rosetta_rots(pose, [8], sfxn, extra_rots=[1, 1, 0, 0], **kw)
   for f in fields:
      assert len(rotdata[f]) == 814
   assert 64 == len(rotdata.onebody)
   t.checkpoint('ex1 ex2')

   print(t)

def test_rosetta_rots(**kw):

   pose = Pose()
   core.pose.make_pose_from_sequence(pose, 'AAAAAAAAAAAAAAA', 'fa_standard', auto_termini=False)
   ala_to_virtCB(pose)

   for i in range(1, pose.size() + 1):
      pose.set_phi(i, -47)
      pose.set_psi(i, -57)
      pose.set_omega(i, 180)
   sfxn = rp.rosetta.get_score_function()
   sfxn.score(pose)
   # pose.dump_pdb("refhelix.pdb").

   fields = [
      'coords', 'rotnum', 'atomnum', 'resname', 'atomname', 'atomtype', 'rosetta_atom_type_index'
   ]

   t = rp.Timer().start()

   rotdata = get_rosetta_rots(pose, [8], sfxn, **kw)
   for f in fields:
      assert len(rotdata[f]) == 105
   assert 9 == len(rotdata.onebody)
   t.checkpoint('no ex')

   rotdata = get_rosetta_rots(pose, [8], sfxn, extra_rots=[1, 0, 0, 0], **kw)
   for f in fields:
      assert len(rotdata[f]) == 294
   assert 24 == len(rotdata.onebody)
   t.checkpoint('ex1')

   rotdata = get_rosetta_rots(pose, [8], sfxn, extra_rots=[0, 1, 0, 0], **kw)
   for f in fields:
      assert len(rotdata[f]) == 287
   assert 23 == len(rotdata.onebody)
   t.checkpoint('ex2')

   rotdata = get_rosetta_rots(pose, [8], sfxn, extra_rots=[1, 1, 0, 0], **kw)
   for f in fields:
      assert len(rotdata[f]) == 814
   assert 64 == len(rotdata.onebody)
   t.checkpoint('ex1 ex2')

   print(t)

if __name__ == '__main__':
   # test_rosetta_rots_1res(dump_rotamers=True)
   # test_rosetta_rots  _1res()
   test_rosetta_rots()
   print('test_rosetta_rots.py DONE')