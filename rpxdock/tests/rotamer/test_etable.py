import numpy as np, rpxdock as rp
from rpxdock.rotamer.etable import (_get_etable, get_etables, make_2res_gly_poses,
                                    earray_rosetta_sfxn, earray_score_2res_pose)
from pyrosetta import pose_from_file
from rpxdock.rosetta.triggers_init import remove_terminus_variants
from rpxdock.rosetta.rosetta_util import xform_pose

def test_get_etables():
   N = 1024
   etables = get_etables(N)
   ch3ch3, ch3hapo, hapohapo = etables[2, 2], etables[2, 3], etables[3, 3]
   c_and_h = _get_etable('Hapo', 'Hapo', N)
   c_and_h_hat = ch3ch3 + 2 * ch3hapo + hapohapo
   assert np.allclose(c_and_h, c_and_h_hat)

   import matplotlib

def scale_raw_score(e):
   s = np.zeros(e.shape)
   s[e > 0] = np.sqrt(e[e > 0])
   s[e < 0] = -np.sqrt(-e[e < 0])
   return s

def test_etable_v_rosetta_2res(Ntest=1, N=100, plot=False):

   #  64             9398230856304585
   # 128 e v s e<3 0.9804708700163653
   # 192             9867697120396796
   #                 9893726064006946
   # 200           0.9878583845912273
   # 228           0.9915802097700429
   # 256 e v s e<3 0.9905069439891698
   #                 9893203434735797
   #                 9917960653654193
   #                 9918968055523946
   # 512 e v s e<3 0.9903913481726961
   #1024           0.9900235243619212
   # 2048          0.993876241929788

   etables = get_etables(N)

   pose = pose_from_file(rp.data.pdbdir + '/twores.pdb')
   remove_terminus_variants(pose)
   assert pose.size() == 2
   gly1pose, gly2pose, gly12pose = make_2res_gly_poses(pose)

   # pose.dump_pdb('fullpose.pdb')
   # gly1pose.dump_pdb('gly1pose.pdb')
   # gly2pose.dump_pdb('gly2pose.pdb')
   # gly12pose.dump_pdb('gly12pose.pdb')

   etot, rtot, ntot = 0, 0, 0
   e, r, s = -12345 * np.ones(Ntest), -12345 * np.ones(Ntest), -12345 * np.ones(Ntest)

   for i in range(Ntest):

      xform = rp.homog.rand_xform_small(1, 0.05, 0.166)
      prevsc = e[i - 1] if i > 0 else 9e9
      prev = pose.clone(), gly1pose.clone(), gly2pose.clone(), gly12pose.clone()
      xform_pose(pose, xform, 2)
      xform_pose(gly1pose, xform, 2)
      xform_pose(gly2pose, xform, 2)
      xform_pose(gly12pose, xform, 2)

      r[i] = earray_rosetta_sfxn(pose)
      gly1score = earray_rosetta_sfxn(gly1pose)
      gly2score = earray_rosetta_sfxn(gly2pose)
      gly12score = earray_rosetta_sfxn(gly12pose)
      glyscore = gly12score + gly1score + gly2score
      s[i] = r[i] - gly1score - gly2score - gly12score

      e[i] = earray_score_2res_pose(pose, etables)
      e[i] += 1e-6  # hack avoids div by 0 errors

      if abs(gly1score) > 0.1 or abs(gly2score) > 0.1 or abs(gly12score) > 0.1:
         pose, gly1pose, gly2pose, gly12pose = prev
         continue

      if s[i] < -8:
         print(e[i], s[i])
         pose.dump_pdb('full.pdb')
         gly1pose.dump_pdb('gly1.pdb')
         gly2pose.dump_pdb('gly2.pdb')
         gly12pose.dump_pdb('gly12.pdb')
         assert 0
      # print(f"HIT")

      # print(e[i] - prevsc, e[i])
      if e[i] - prevsc > 0 and np.random.random() < 0.1:
         pose, gly1pose, gly2pose, gly12pose = prev

      if (i + 1) % 50 == 0:
         print(f'| {e[i]:7.3f} | {s[i]:7.3f} {e[i]/s[i]:7.3f} | {r[i]:7.3f} {e[i]/r[i]:7.3f} |')

      etot += e[i]
      rtot += r[i]
      ntot += 1
      print('.', end='')

   e2 = e[e != -12345]
   r2 = r[e != -12345]
   s2 = s[e != -12345]

   print('AVG', np.sqrt(abs(etot)) / np.sqrt(abs(rtot)), 'N', ntot)
   print('e v r ALL', np.corrcoef(np.stack([e2, r2]))[0, 1])
   print('e v s ALL', np.corrcoef(np.stack([e2, s2]))[0, 1])

   ethresh = 6
   idx = np.logical_and(e2 != -12345,
                        np.logical_and(r2 < 10, np.logical_and(e2 < ethresh, s2 < ethresh)))
   e3 = e2[idx]
   r3 = r2[idx]
   s3 = s2[idx]

   print('nsamp', len(e3))
   print('e v r e<3', np.corrcoef(np.stack([e3, r3]))[0, 1])
   print('e v s e<3', np.corrcoef(np.stack([e3, s3]))[0, 1])

   if plot:
      import sys
      sys.stdout.flush()
      toplot = dict(etable=e3, rosetta=s3)  #, rosetta_raw=r3)
      rp.util.plot.coplot(toplot)

if __name__ == '__main__':
   test_get_etables()
   test_etable_v_rosetta_2res()
   # test_etable_v_rosetta_2res(Ntest=1000, N=256, plot=True)
