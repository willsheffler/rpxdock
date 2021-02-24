import rpxdock as rp, numpy as np
from rpxdock.rosetta.triggers_init import create_residue, Pose, AtomID
from pyrosetta.rosetta.numeric import xyzVector_double_t as xyzVec
from pyrosetta.rosetta.core.scoring import ScoreFunction, ScoreType, get_score_function
from rpxdock.rosetta.triggers_init import create_residue

earray_rosetta_sfxn = get_score_function()
# earray_rosetta_sfxn = ScoreFunction()
# earray_rosetta_sfxn.set_weight(ScoreType.fa_atr, 1.0)
# earray_rosetta_sfxn.set_weight(ScoreType.fa_rep, 0.55)
# earray_rosetta_sfxn.set_weight(ScoreType.fa_sol, 1.0)
# earray_rosetta_sfxn.set_weight(ScoreType.fa_elec, 1.0)

# # beta_nov16
# #   beta energy function following parameter refitting (Frank DiMaio and Hahnbeom Park), November 2016
# #METHOD_WEIGHTS ref 1.8394 3.6196 -2.3716 -2.7348 1.0402 0.83697 -0.45461 0.73287 -1.5107 0.18072 0.60916 -0.93687 -2.4119 -0.18838 -1.2888 -0.77834 -1.0874 1.9342 1.6906 1.2797
# METHOD_WEIGHTS ref 2.3386 3.2718 -2.2837 -2.5358 1.4028 1.2108 0.134426 1.0317 -1.6738 0.729516 1.2334 -0.873554 -5.1227 -1.0644 -1.281 -1.1772 -1.425 2.085 3.035 0.964136
# fa_atr 1
# fa_rep 0.55
# fa_sol 1.0
# fa_intra_atr_xover4 1
# fa_intra_rep_xover4 0.55
# fa_intra_sol_xover4 1
# lk_ball 0.92
# lk_ball_iso -0.38
# lk_ball_bridge -0.33
# lk_ball_bridge_uncpl -0.33
# fa_elec 1.0
# fa_intra_elec 1.0
# pro_close 1.25
# hbond_sr_bb 1.0
# hbond_lr_bb 1.0
# hbond_bb_sc 1.0
# hbond_sc 1.0
# dslf_fa13 1.25
# rama_prepro 0.50
# omega 0.48
# p_aa_pp 0.61
# fa_dun_rot 0.76
# fa_dun_dev 0.69
# fa_dun_semi 0.78
# hxl_tors 1.0
# gen_bonded 1.0
# ref 1

def two_atom_pose(a1, a2):
   pose = Pose()
   pose.append_residue_by_jump(create_residue(a1), 1)
   pose.append_residue_by_jump(create_residue(a2), 1)
   return pose

def set_2atom_dist(pose, dist):
   for i in range(pose.residue(1).natoms()):
      pose.set_xyz(AtomID(i + 1, 1), xyzVec(0, 0, 0))
   for i in range(pose.residue(2).natoms()):
      pose.set_xyz(AtomID(i + 1, 2), xyzVec(dist, 0, 0))
   return pose

def atom_atom_score(pose, dist):
   set_2atom_dist(pose, dist)
   return earray_rosetta_sfxn.score(pose)

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

def earray_d2resl(nsamp):
   return (nsamp - 1) / 36.0

# def get_etable(a1, a2, nsamp):
#    pose = two_atom_pose(a1, a2)
#    d2resl = (nsamp - 1) / 36.0
#    r = [atom_atom_score(pose, np.sqrt(d2 / d2resl)) for d2 in range(0, nsamp)]
#    # pose.dump_pdb('test.pdb')
#    return np.array(r, dtype='f4')

def _get_etable(an1, an2, nsamp):
   e = two_atom_pose(an1, an2)
   d2resl = (nsamp - 1) / 36.0
   samps = range(0, nsamp)
   i2d = lambda i: np.sqrt((i + 0.5) / d2resl)
   return np.array([atom_atom_score(e, i2d(i)) for i in samps], dtype='f4')

def get_etables(nsamp, debug=True):
   d2resl = (nsamp - 1) / 36.0
   samps = range(0, nsamp)

   #    pose_ch3 = two_atom_pose('CH3', 'CH3')
   #    ch3_ch3 = np.array([atom_atom_score(pose_ch3, np.sqrt(d2 / d2resl)) for d2 in samps],
   #                       dtype='f4')
   #
   #    pose_ch3hapo = two_atom_pose('CH3', 'Hapo')
   #    ch3_hap = np.array([atom_atom_score(pose_ch3hapo, np.sqrt(d2 / d2resl)) for d2 in samps],
   #                       dtype='f4')
   #    ch3_hap = ch3_hap - ch3_ch3
   #
   #    pose_hapo = two_atom_pose('Hapo', 'Hapo')
   #    hap_hap = np.array([atom_atom_score(pose_hapo, np.sqrt(d2 / d2resl)) for d2 in samps],
   #                       dtype='f4')
   #    hap_hap = hap_hap - ch3_ch3 - 2 * ch3_hap

   ch1_ch1 = _get_etable('CH1', 'CH1', nsamp)
   ch1_ch2 = _get_etable('CH1', 'CH2', nsamp)
   ch1_ch3 = _get_etable('CH1', 'CH3', nsamp)
   ch2_ch2 = _get_etable('CH2', 'CH2', nsamp)
   ch2_ch3 = _get_etable('CH2', 'CH3', nsamp)
   ch3_ch3 = _get_etable('CH3', 'CH3', nsamp)
   ch3_hap = _get_etable('CH3', 'Hapo', nsamp) - ch3_ch3
   hap_hap = _get_etable('Hapo', 'Hapo', nsamp) - ch3_ch3 - 2 * ch3_hap

   # np.set_printoptions(formatter=dict(float=lambda x: '%7.3f' % x))
   # print(np.stack([ch1_ch1, ch1_ch2, ch1_ch3, ch2_ch2, ch2_ch3, ch3_ch3], axis=1)[0:nsamp:50])

   etables = dict()
   etables[0, 0] = ch1_ch1
   etables[0, 1] = ch1_ch2
   etables[0, 2] = ch1_ch3
   etables[0, 3] = ch3_hap

   etables[1, 0] = ch1_ch2
   etables[1, 1] = ch2_ch2
   etables[1, 2] = ch2_ch3
   etables[1, 3] = ch3_hap

   etables[2, 0] = ch1_ch3
   etables[2, 1] = ch2_ch3
   etables[2, 2] = ch3_ch3
   etables[2, 3] = ch3_hap

   etables[3, 0] = ch3_hap
   etables[3, 1] = ch3_hap
   etables[3, 2] = ch3_hap
   etables[3, 3] = hap_hap

   r = earray_r(etables[0, 0])

   return etables

def make_2res_gly_poses(pose):
   gly1pose, gly2pose, gly12pose = pose.clone(), pose.clone(), pose.clone()
   gly1pose.replace_residue(1, create_residue('GLY'), True)
   gly2pose.replace_residue(2, create_residue('GLY'), True)
   gly12pose.replace_residue(1, create_residue('GLY'), True)
   gly12pose.replace_residue(2, create_residue('GLY'), True)
   return gly1pose, gly2pose, gly12pose

# etable C-C  CB   CG1 894 -0.031581532
# etable C-C  CB   CG2 965 -0.006509287
# etable C-C  CG   CB  686 -0.19163097
# etable C-C  CG   CG1 514 -0.37222672
# etable C-C  CG   CG2 617 -0.26202902
# etable C-C  CD1  CB  727 -0.15228297
# etable C-C  CD1  CG1 639 -0.23926625
# etable C-C  CD1  CG2 666 -0.21165727
# etable C-C  CD2  CB  424 -0.39467198
# etable C-C  CD2  CG1 323 -0.39467198
# etable C-C  CD2  CG2 318 -0.39467198
#
def earray_score_2res_pose(pose, etables, debug=True):

   rosetta_atom_type_to_rpx = -np.ones(999, dtype='i4')
   rosetta_atom_type_to_rpx[4] = 0
   rosetta_atom_type_to_rpx[5] = 1
   rosetta_atom_type_to_rpx[6] = 2
   rosetta_atom_type_to_rpx[32] = 3
   cat = [4, 5, 6]
   hat = [32]

   N = len(etables[0, 0])
   d2resl = earray_d2resl(N)
   etot = 0
   for ia in range(1, pose.residue(1).natoms() + 1):
      at1 = rosetta_atom_type_to_rpx[pose.residue(1).atom_type_index(ia)]
      if at1 < 0: continue
      an1 = pose.residue(1).atom_name(ia)
      xyz1 = pose.residue(1).xyz(ia)
      # print(pose.residue(1).atom_name(ia), at1, pose.residue(1).atom_type_index(ia))
      for ja in range(1, pose.residue(2).natoms() + 1):
         at2 = rosetta_atom_type_to_rpx[pose.residue(2).atom_type_index(ja)]
         if at2 < 0: continue
         # print(at1, at2)
         an2 = pose.residue(2).atom_name(ja)
         xyz2 = pose.residue(2).xyz(ja)
         d2 = xyz1.distance_squared(xyz2)
         d2idx = int(d2 * d2resl)
         if d2idx >= N:
            continue

         e = etables[at1, at2][d2idx]

         # e1 = etables[at1, at2][d2idx]
         # e2 = etables[at1, at2][d2idx + 1]
         # e = e1
         # if e != 0:
         # print(at1, an1, at2, an2, e)

         etot += e

   return etot
