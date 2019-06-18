import xarray
from rpxdock.rotamer.richardson import get_rotamer_space, boundsfields
import numpy as np

def test_richardson_space():
   rotspace = get_rotamer_space(disulf=False, concat=False)
   print(rotspace.to_dataframe().shape)
   # assert rotspace.to_dataframe().shape == (163 - 10, 29 - 3 * 4 + 1)
   assert rotspace.to_dataframe().shape == (153, 19)
   # print(rotspace.loc[:, "lb1 lb2 lb3 lb4 nchi".split()])
   # print(rotspace.loc[list('DNEQ'),
   # 'lb1 ub1 lb2 ub2 lb3 ub3 lb4 ub4'.split()])
   # for k, g in rotspace.loc['D'].groupby(['lb1', 'ub1']):
   # print(g.loc[:, boundsfields])

def test_richardson_space_cache():
   assert get_rotamer_space() is get_rotamer_space()
   assert get_rotamer_space(disulf=1) is get_rotamer_space(disulf=1)
   assert get_rotamer_space(disulf=1) is not get_rotamer_space(disulf=0)

def test_richardson_space_nchi():
   for _nchi, group in get_rotamer_space().groupby("nchi"):
      nchi = int(_nchi)
      assert nchi == _nchi
      for i in range(1, nchi + 1):
         assert np.alltrue(group["lb%d" % i].notnull())
      for i in range(nchi + 1, 5):
         assert np.alltrue(group["lb%d" % i].isnull())

def test_richardson_space_concat():
   rotspace = get_rotamer_space(concat=False)
   rotspace_concat = get_rotamer_space(concat=True)
   # print("====================================================")
   # print(rotspace.loc[list("DNEQ"), boundsfields])
   # print("================ concat ============================")
   # print(rotspace_concat.loc[list("DNEQ"), boundsfields])
   assert rotspace_concat.to_dataframe().shape[0] < rotspace.to_dataframe().shape[0]

if __name__ == "__main__":
   test_richardson_space()
   test_richardson_space_cache()
   test_richardson_space_nchi()
   test_richardson_space_concat()
