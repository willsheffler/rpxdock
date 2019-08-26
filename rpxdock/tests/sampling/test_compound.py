import numpy as np, pytest, rpxdock as rp
from rpxdock.sampling import *

ch2 = CartHier1D_f8([0], [2], [2])
ch3 = CartHier1D_f8([0], [3], [3])
ch5 = CartHier1D_f8([0], [5], [5])
h1 = CompoundHier(ch2, ch3, ch5)

ch2d = CartHier2D_f8([0, 0], [1, 1], [1, 1])
ch3d = CartHier3D_f8([0, 0, 0], [1, 1, 1], [1, 1, 1])
h2 = CompoundHier(ch2d, ch3d)

ch2d3 = CartHier2D_f8([0, 0], [3, 3], [3, 3])
ch3d2 = CartHier3D_f8([0, 0, 0], [2, 2, 2], [2, 2, 2])
h3 = CompoundHier(ch2d3, ch3d2)

def test_compound_single():
   h = CompoundHier(ch2)
   for resl in range(3):
      assert h.size(resl) == ch2.size(resl)
      ok, hx = h.get_xforms(resl, np.arange(h.size(resl)))
      ok, cx = ch2.get_xforms(resl, np.arange(ch2.size(resl)))
      assert np.allclose(hx[:, 0], cx)

def test_compound_ncell_dim():
   assert np.all(h1.ncells == [2, 3, 5])
   assert h1.ncell == 30
   assert h1.dim == 3

   assert np.all(h2.ncells == [1, 1])
   assert h2.ncell == 1
   assert h2.dim == 5

   assert np.all(h3.ncells == [9, 8])
   assert h3.ncell == 72
   assert h3.dim == 5

def test_compound_basic_indexing():
   with pytest.raises(ValueError):
      h1.check_indices(0, range(100))
   assert np.all(h1.cell_index_of(0, range(30)) == range(30))
   assert np.all(h1.cell_index_of(1, range(30)) == np.repeat(range(4), 8)[:30])
   assert np.all(h1.hier_index_of(0, range(30)) == 0)
   assert np.all(h1.hier_index_of(1, range(30)) == np.tile(range(8), 4)[:30])

   assert np.all(h3.cell_index_of(0, range(72)) == range(72))
   assert np.all(h3.cell_index_of(1, range(999)) == np.repeat(range(72), 32)[:999])
   assert np.all(h3.hier_index_of(0, range(72)) == 0)
   assert np.all(h3.hier_index_of(1, range(999)) == np.tile(range(32), 72)[:999])

def test_compound_split_cell():
   ref = np.array([
      [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
      [0, 0, 1, 1, 2, 2, 0, 0, 1, 1, 2, 2, 0, 0, 1, 1, 2, 2, 0, 0, 1, 1, 2, 2, 0, 0, 1, 1, 2, 2],
      [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4],
   ])
   for resl in range(5):
      split = h1.split_indices_cell(resl, range(0, h1.size(resl), h1.cellsize(resl)))
      assert np.all(split == ref)

   for resl in range(5):
      split = h3.split_indices_cell(resl, range(0, h3.size(resl), h3.cellsize(resl)))
      assert np.all(split == [np.tile(range(9), 8), np.repeat(range(8), 9)])

def test_compound_split_hier():
   split = h2.split_indices_hier(1, np.arange(h2.cellsize(1)))
   assert np.all(split[0] == np.tile(range(4), 8))
   assert np.all(split[1] == np.repeat(range(8), 4))

   split = h3.split_indices_hier(1, np.arange(h3.cellsize(1)))
   assert np.all(split[0] == np.tile(range(4), 8))
   assert np.all(split[1] == np.repeat(range(8), 4))

def test_compound_split():
   resl = 0
   # splitcell = h3.split_indices_cell(resl, np.arange(h3.size(resl)))
   # splithier = h3.split_indices_hier(resl, np.arange(h3.size(resl)))
   split = h3.split_indices(resl, np.arange(h3.size(resl)))
   assert np.max(split[0]) + 1 == ch2d3.size(resl)
   assert np.max(split[1]) + 1 == ch3d2.size(resl)
   # print(split)

def test_compound_get_xforms():
   ok, a = h3.parts[0].get_xforms(0, [0])
   ok, b = h3.get_xforms(0, [0])
   assert np.allclose(a, b[0, 0])

   for resl in range(3):
      valid, xforms = h3.get_xforms(resl, range(h3.size(resl)))
      assert xforms.shape == (h3.size(resl), 2, 4, 4)

      _, x0 = h3.parts[0].get_xforms(resl, range(h3.parts[0].size(resl)))
      _, x1 = h3.parts[1].get_xforms(resl, range(h3.parts[1].size(resl)))

      u0 = np.unique(xforms[:, 0, :3, 3], axis=0)
      u0b = np.unique(x0[:, :3, 3], axis=0)
      assert np.allclose(u0, u0b)

      u1 = np.unique(xforms[:, 1, :3, 3], axis=0)
      u1b = np.unique(x1[:, :3, 3], axis=0)
      assert np.allclose(u1, u1b)

def test_product_hier():
   g = ZeroDHier(rp.homog.rand_xform())
   h = ProductHier(ch2, g)
   for resl in range(3):
      assert h.size(resl) == ch2.size(resl)
      ok, hx = h.get_xforms(resl, np.arange(h.size(resl)))
      ok, cx = ch2.get_xforms(resl, np.arange(ch2.size(resl)))
      assert np.allclose(hx, cx @ g.samples[0])

   g = ZeroDHier(rp.homog.rand_xform(3))
   assert g.ncell == 3
   h = ProductHier(ch2, g)
   for resl in range(3):
      assert h.size(resl) == 3 * ch2.size(resl)
      ok, hx = h.get_xforms(resl, np.arange(h.size(resl)))
      ok, cx = ch2.get_xforms(resl, np.arange(ch2.size(resl)))
      for i in range(g.ncell):
         for j in range(ch2.size(resl)):
            k = ch2.size(resl) * i + j
            assert np.allclose(hx[k], cx[j] @ g.samples[i])

   assert g.ncell == 3
   g = ZeroDHier(np.stack([np.eye(4), np.eye(4), np.eye(4)]))
   h = ProductHier(g, ch2)
   for resl in range(3):
      assert h.size(resl) == 3 * ch2.size(resl)
      ok, hx = h.get_xforms(resl, np.arange(h.size(resl)))
      ok, cx = ch2.get_xforms(resl, np.arange(ch2.size(resl)))
      for ic in range(ch2.ncell):
         for jc in range(g.ncell):
            for kh in range(2**resl):
               idx = g.ncell * 2**resl * ic + 2**resl * jc + kh
               assert np.allclose(hx[idx], g.samples[jc] @ cx[ic * 2**resl + kh])

def test_compound_product_hier():
   g1 = ZeroDHier(rp.homog.rand_xform())
   g2 = ZeroDHier(rp.homog.rand_xform())
   p1 = ProductHier(ch2, g1)
   p2 = ProductHier(ch2, g2)
   h = CompoundHier(p1, p2)
   assert h.ncell == 4
   ok, x1 = p1.get_xforms()
   ok, x2 = p2.get_xforms()
   ok, x = h.get_xforms()
   assert np.all(ok)
   assert len(x) is 4
   assert np.allclose(x[0, 0], x1[0])
   assert np.allclose(x[0, 1], x2[0])
   assert np.allclose(x[1, 0], x1[1])
   assert np.allclose(x[1, 1], x2[0])
   assert np.allclose(x[2, 0], x1[0])
   assert np.allclose(x[2, 1], x2[1])
   assert np.allclose(x[3, 0], x1[1])
   assert np.allclose(x[3, 1], x2[1])

if __name__ == '__main__':
   test_compound_single()
   test_compound_ncell_dim()
   test_compound_basic_indexing()
   test_compound_split_cell()
   test_compound_split_hier()
   test_compound_split()
   test_compound_get_xforms()
   test_product_hier()
   test_compound_product_hier()
