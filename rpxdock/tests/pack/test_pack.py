import os
import rpxdock as rp
from willutil import Bunch
import numpy as np
from rpxdock.rosetta.triggers_init import pose_from_file
from rpxdock.rotamer import rosetta_rots
from pyrosetta.rosetta.core.pack import pack_rotamers
from rpxdock.rosetta.triggers_init import create_residue, Pose, AtomID
from willutil import Timer

# _iface2_ires = [x - 23 for x in (24, 25, 27, 28, 31, 32, 34, 35, 38, 39, 41, 42)]

# _iface_ires = [
#    2, 3, 5, 6, 9, 10, 12, 13, 16, 17, 19, 20, 23, 24, 25, 27, 28, 31, 32, 34, 35, 38, 39, 41, 42
# ]
# _iface1_ires = [2, 3, 5, 6, 9, 10, 12, 13, 16, 17, 19, 20, 23]
#
# _iface2_ires = [1, 2, 4, 5, 8, 9, 11, 12, 15, 16, 18, 19]
# _iface2_ires = [1, 2]

# _iface_ires = list(range(1, 45))
# _iface1_ires = list(range(1, 24))
# _iface2_ires = list(range(1, 22))

_iface_ires = 'all'
_iface1_ires = 'all'
_iface2_ires = 'all'

_ex = [1, 1, 0, 0]
# _ex = [1, 1, 0, 0]

def build_packing_data(poseA, poseB, **kw):
   # if False or os.path.exists('testpackdata.pickle'):
   # rotdata1, rotdata2 = rp.load('testpackdata.pickle')
   # else:

   rotdata1 = rosetta_rots.get_rosetta_rots(poseA, whichres=_iface1_ires, extra_rots=_ex, **kw)
   rotdata2 = rosetta_rots.get_rosetta_rots(poseB, whichres=_iface2_ires, extra_rots=_ex, **kw)
   print('nrot atoms', len(rotdata1.coords), len(rotdata2.coords))

   ids_a = np.arange(len(rotdata1.coords))
   ids_b = np.arange(len(rotdata2.coords))
   bvh_a = rp.BVH(rotdata1.coords, [], ids_a)
   bvh_b = rp.BVH(rotdata2.coords, [], ids_b)
   # pairsAA, _ = rp.bvh.bvh_collect_pairs_vec(bvh_a, bvh_a, np.eye(4), np.eye(4), 6.0)
   # pairsBB, _ = rp.bvh.bvh_collect_pairs_vec(bvh_b, bvh_b, np.eye(4), np.eye(4), 6.0)
   # print('bvh sizes ', len(bvh_a), len(bvh_b))

   t = Timer().start()
   pairsAB, _ = rp.bvh.bvh_collect_pairs_vec(bvh_a, bvh_b, np.eye(4), rp.homog.htrans([0, 0, 0]),
                                             6.0)
   t.checkpoint('bvh get atom pairs ')

   print(t.report(scale=1000, precision='10.3f'))
   # print('n atom pairs', len(pairsAA), len(pairsAB), len(pairsBB))
   print('n atom pairs', len(pairsAB))

def mutate_to_ala(pose):
   pose = pose.clone()
   for i in range(1, pose.size() + 1):
      pose.replace_residue(i, create_residue('ALA'), True)
   return pose

def select_by_sasa(pose, thresh):
   pass

def rosetta_pack_test(pose):

   sfxn = rp.rosetta.get_score_function()
   pose = mutate_to_ala(pose)
   task = rosetta_rots.create_rosetta_packer_task(pose, _iface_ires, extra_rots=_ex)

   # pose.dump_pdb('pack_before.pdb')
   t = Timer().start()
   pack_rotamers(pose, sfxn, task)
   t.checkpoint('rosetta pack iface')
   print(t.report(scale=1000, precision='10.3f'))
   # pose.dump_pdb('pack_after.pdb')

def check_hash_vs_array():
   pose1 = pose_from_file(rp.data.pdbdir + '/zipper21.pdb.gz')
   pose2 = pose_from_file(rp.data.pdbdir + '/zipper22.pdb.gz')
   body1 = rp.Body(pose1)
   body2 = rp.Body(pose2)

   rots = rosetta_rots.get_helix_rotamers()

def perf_hash_vs_array():
   '''
   ML sequence design Ustice(Sp?) David Jorgen
   '''

   Nsamp = int(1e6)
   Nfull = int(1e6)
   Nrel = int(1e4)

   t = Timer().start()

   nonzero = np.random.randint(0, Nfull, Nrel, dtype='u8')
   check = nonzero[np.random.randint(0, Nrel, Nsamp, dtype='u8')]
   checkb = np.random.randint(0, Nfull / 1000, Nsamp, dtype='u8')
   # check = np.random.randint(0, Nfull, Nsamp, dtype='u8')

   a = np.zeros(Nfull, dtype='f4')
   b = np.zeros(int(Nfull / 1000), dtype='f4')
   p = rp.phmap.PHMap_u8u8()

   a[nonzero] = 1
   p[nonzero] = 1

   t.checkpoint('setup')
   _ = p[check]
   t.checkpoint('phmap')

   _ = a[check]
   _ = b[checkb]
   t.checkpoint('array')

   print(t)

if __name__ == '__main__':

   pose = pose_from_file(rp.data.pdbdir + '/medium_iface.pdb.gz')
   rosetta_pack_test(pose)

   # check_hash_vs_array()
   pose1 = pose_from_file('rpxdock/data/pdb/medium_ifaceA.pdb.gz')
   pose2 = pose_from_file('rpxdock/data/pdb/medium_ifaceB.pdb.gz')
   print('pose 1/2 nres', pose1.size(), pose2.size())

   build_packing_data(pose1, pose2)
