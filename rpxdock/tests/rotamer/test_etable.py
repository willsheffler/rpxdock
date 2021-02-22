from rpxdock.rotamer.etable import *
from pyrosetta import pose_from_file
from rpxdock.rosetta.triggers_init import remove_terminus_variants
from rpxdock.rosetta.rosetta_util import xform_pose

def test_etable_a1a2(aname1, aname2):
   N = 64

   print(f"======= {aname1} === {aname2} =======")
   etable = get_etable(aname1, aname2, N)
   assert len(etable) == N

   slope = earray_slope(etable)
   assert slope.shape == etable.shape
   edelta = etable[1:] - etable[:-1]
   # print(slope)
   print('edelta')
   edelta = edelta
   q = np.quantile(edelta, np.arange(5) / 4)
   for i in range(len(q)):
      print(q[i])
   r = earray_r(etable)
   for i in range(1, N):
      print(f"{r[i]:7.3f}     {etable[i]-etable[i-1]:7.3f}   {etable[i]:7.3f}")
   # MIN_PRECISION = 0.1
   # assert np.all(np.abs(etable[65:] - etable[64:-1]) < MIN_PRECISION)

def test_get_etables():
   N = 512
   ch3ch3, ch3hapo, hapohapo = get_etables(N)

   c_and_h = get_etable('Hapo', 'Hapo', N)
   assert np.allclose(c_and_h, ch3ch3 + 2 * ch3hapo + hapohapo)

   import matplotlib

def scale_raw_score(e):
   s = np.zeros(e.shape)
   s[e > 0] = np.sqrt(e[e > 0])
   s[e < 0] = -np.sqrt(-e[e < 0])
   return s

def test_etable_v_rosetta_2res():
   Ntest = 1000

   N = 8192
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
   e, r, s = np.zeros(Ntest), -np.ones(Ntest), np.zeros(Ntest)

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

   e2 = e
   r2 = r
   s2 = s

   print('AVG', np.sqrt(abs(etot)) / np.sqrt(abs(rtot)), 'N', ntot)
   print('e v r ALL', np.corrcoef(np.stack([e2, r2]))[0, 1])
   print('e v s ALL', np.corrcoef(np.stack([e2, s2]))[0, 1])

   idx = np.logical_and(r2 < 10, np.logical_and(e2 < 3, s2 < 3))
   e3 = e2[idx]
   r3 = r2[idx]
   s3 = s2[idx]

   toplot = dict(etable=e3, rosetta=s3, rosetta_raw=r3)
   rp.util.plot.coplot(toplot)

   print('e v r e<3', np.corrcoef(np.stack([e3, r3]))[0, 1])
   print('e v s e<3', np.corrcoef(np.stack([e3, s3]))[0, 1])

   # energy_graph = pose.energies().energy_graph()
   # eweights = pose.energies().weights()
   # nonbonded_energy = 0
   # for i in range(1, 3):
   #    for j in range(1, 3):
   #       edge = energy_graph.find_edge(i, j)
   #       if not edge:
   #          print('res-res energy', i, j, 'NONE')
   #          pass
   #       else:
   #          e = edge.dot(eweights)
   #          print('res-res energy', i, j, e)
   #          nonbonded_energy += e

   # assert 0

if __name__ == '__main__':
   # test_etable_a1a2('CH3', 'CH3')
   # test_etable_a1a2('CH3', 'Hapo')
   # test_etable_a1a2('Hapo', 'Hapo')
   # test_get_errays()
   test_etable_v_rosetta_2res()
