import rpxdock as rp, numpy as np
from rpxdock.rosetta.triggers_init import create_residue, Pose, AtomID, sfxn
from pyrosetta.rosetta.numeric import xyzVector_double_t as xyzVec

def two_atom_pose(a1, a2):
   pose = Pose()
   pose.append_residue_by_jump(create_residue(a1), 1)
   pose.append_residue_by_jump(create_residue(a2), 1)
   return pose

def set_2atom_dist(pose, dist):
   assert pose.size() == 2
   for i in range(pose.residue(1).natoms()):
      pose.set_xyz(AtomID(i + 1, 1), xyzVec(0, 0, 0))
   for i in range(pose.residue(2).natoms()):
      pose.set_xyz(AtomID(i + 1, 2), xyzVec(dist, 0, 0))
   return pose

def atom_atom_score(pose, dist):
   set_2atom_dist(pose, dist)
   return sfxn.score(pose)

def earray_r(e):
   d2resl = (len(e) - 1) / 36.0
   r = np.sqrt(np.arange(len(e)) / d2resl)
   return r

def earray_i(r):
   d2resl = (len(e) - 1) / 36.0
   d2 = r * r / d2resl
   return int(d2)

def earray_slope(e):
   r = earray_r(e)
   dedr = np.zeros(len(e))
   for i in range(1, len(e) - 1):
      de1 = e[i] - e[i - 1]
      dr1 = r[i] - r[i - 1]
      de2 = e[i + 1] - e[i]
      dr2 = r[i + 1] - r[i]
      dedr[i] = (de1 / dr1 + de2 / dr2) * 0.5
   dedr[0] = dedr[1]
   dedr[-1] = dedr[-2]
   return dedr

def get_earray(a1, a2, nsamp):
   pose = two_atom_pose(a1, a2)
   d2resl = (nsamp - 1) / 36.0
   r = [atom_atom_score(pose, np.sqrt(d2 / d2resl)) for d2 in range(0, nsamp)]
   # pose.dump_pdb('test.pdb')
   return np.array(r, dtype='f4')

def get_earrays(nsamp, debug=True):
   d2resl = (nsamp - 1) / 36.0
   samps = range(0, nsamp)

   pose_ch3 = two_atom_pose('CH3', 'CH3')
   ch3ch3 = np.array([atom_atom_score(pose_ch3, np.sqrt(d2 / d2resl)) for d2 in samps],
                     dtype='f4')

   pose_ch3hapo = two_atom_pose('CH3', 'Hapo')
   ch3hapo = np.array([atom_atom_score(pose_ch3hapo, np.sqrt(d2 / d2resl)) for d2 in samps],
                      dtype='f4')
   ch3hapo = ch3hapo - ch3ch3

   pose_hapo = two_atom_pose('Hapo', 'Hapo')
   hapohapo = np.array([atom_atom_score(pose_hapo, np.sqrt(d2 / d2resl)) for d2 in samps],
                       dtype='f4')
   hapohapo = hapohapo - ch3ch3 - 2 * ch3hapo

   r = earray_r(ch3ch3)
   np.set_printoptions(formatter=dict(float=lambda x: '%7.3f' % x))
   print(np.stack([r, ch3ch3, ch3hapo, hapohapo]).T)

   if debug:
      test = two_atom_pose('Hapo', 'CH3')  # backwards!
      test = np.array([atom_atom_score(test, np.sqrt(d2 / d2resl)) for d2 in samps], dtype='f4')
      test = test - ch3ch3
      assert np.allclose(test, ch3hapo)

   return dict(ch3_ch3=ch3ch3, ch3_hapo=ch3hapo, hapo_hapo=hapohapo)
