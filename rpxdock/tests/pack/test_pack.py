import rpxdock as rp, numpy as np
from pyrosetta import pose_from_file

def check_hash_vs_array():
   pose1 = pose_from_file(rp.data.pdbdir + '/zipper1.pdb.gz')
   pose2 = pose_from_file(rp.data.pdbdir + '/zipper2.pdb.gz')
   body1 = rp.Body(pose1)
   body2 = rp.Body(pose2)

   rots = rp.rotamer.rosetta_rots.get_helix_rotamers()

def perf_hash_vs_array():
   '''
   ML sequence design Ustice(Sp?) David Jorgen

    '''

   Nsamp = int(1e6)
   Nfull = int(1e6)
   Nrel = int(1e4)

   t = rp.Timer().start()

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
   check_hash_vs_array()
