import os
import rpxdock as rp
from rpxdock.rosetta.rosetta_util import *
from rpxdock.data import pdbdir

def test_get_sc_coords(top7):
   anames, acoords = get_sc_coords(top7)
   assert len(anames) == len(acoords) == top7.size()
   for a, c in zip(anames, acoords):
      assert len(a) == len(c)
   assert anames[7] == [
      ' N  ', ' CA ', ' C  ', ' O  ', ' CB ', ' CG1', ' CG2', ' CD1', ' H  ', ' HA ', ' HB ',
      '1HG1', '2HG1', '1HG2', '2HG2', '3HG2', '1HD1', '2HD1', '3HD1'
   ]
   assert acoords[23].shape == (15, 4)
   assert np.allclose(acoords[23][:, 3], 1)
   assert np.allclose(
      acoords[23],
      [[2.557, 13.803, 2.939, 1], [3.028, 13.222, 1.686, 1], [4.543, 13.165, 1.862, 1],
       [5.269, 12.601, 1.043, 1], [2.653, 14.08, 0.473, 1], [3.144, 13.491, -0.85, 1],
       [2.474, 14.106, -2.065, 1], [2.501, 15.346, -2.193, 1], [1.926, 13.346, -2.892, 1],
       [2.03230274, 13.22710015, 3.58175357, 1], [2.56067975, 12.2447251, 1.56050543, 1],
       [1.56929971, 14.18810902, 0.42478765, 1], [3.07664604, 15.07775054, 0.58810302, 1],
       [4.21952131, 13.64757636, -0.92852648, 1], [2.96013644, 12.41752788, -0.84807989, 1]])

   # aset = set()
   # for a in anames:
   #    aset.update(a)
   # print(aset)

def test_get_trimmed_poses(dhr14):

   print(os.getcwd())
   os.system('rm nterm?.pdb')
   os.system('rm cterm?.pdb')

   n, c = get_trimmed_poses(
      dhr14,
      helix_trim_max=5,
      helix_trim_nres_ignore_end=3,
      individual_pieces=False,
      trim_from_N=True,
      trim_from_C=True,
   )
   # for i, p in enumerate(n):
   # p.dump_pdb('nterm%i.pdb' % i)
   # for i, p in enumerate(c):
   # print(p.size())
   # p.dump_pdb('cterm%i.pdb' % i)

   nlen = [316, 297, 277, 257, 237, 217]
   clen = [314, 294, 274, 254, 234, 217]
   assert [p.size() for p in n] == nlen
   assert [p.size() for p in c] == clen

   nseq = 'NERVKQLAEK VIEIVKELAE VNEIVKQLAE VIYIVKILAE VNEIVKQLAE VIYIVKILAE'.split()
   cseq = 'EHIEKILEEL NEIVKQLEEV IYIVKILAEL NEIVKQLAEV IYIVKILAEL VKQLAEVAKE'.split()
   assert [p.sequence()[:10] for p in n] == nseq
   assert [p.sequence()[-10:] for p in c] == cseq

if __name__ == '__main__':
   from rpxdock.rosetta.triggers_init import get_pose_cached

   # test_get_sc_coords(get_pose_cached('top7.pdb.gz', pdbdir))
   test_get_trimmed_poses(get_pose_cached('DHR14.pdb.gz', pdbdir))
   # test_get_trimmed_poses(get_pose_cached('top7.pdb.gz', pdbdir))
   print('DONE')
