from rpxdock.rotamer.earray import *
from pyrosetta import pose_from_file

def test_earray_a1a2(aname1, aname2):
   N = 512

   print(f"======= {aname1} === {aname2} =======")
   earray = get_earray(aname1, aname2, N)
   assert len(earray) == N

   slope = earray_slope(earray)
   assert slope.shape == earray.shape
   edelta = earray[1:] - earray[:-1]
   # print(slope)
   print('edelta')
   edelta = edelta
   q = np.quantile(edelta, np.arange(5) / 4)
   for i in range(len(q)):
      print(q[i])
   r = earray_r(earray)
   for i in range(1, N):
      print(f"{r[i]:7.3f}     {earray[i]-earray[i-1]:7.3f}   {earray[i]:7.3f}")
   # MIN_PRECISION = 0.1
   # assert np.all(np.abs(earray[65:] - earray[64:-1]) < MIN_PRECISION)

def test_earray_hapo():
   pass

def test_get_errays():
   ea = get_earrays(32)

def test_earray_v_rosetta_2res():
   pose = pose_from_file(rp.data.pdbdir + '/twores.pdb')
   print(pose)

if __name__ == '__main__':
   # test_earray_a1a2('CH3', 'CH3')
   # test_earray_a1a2('CH3', 'Hapo')
   # test_earray_a1a2('Hapo', 'Hapo')
   test_get_errays()
