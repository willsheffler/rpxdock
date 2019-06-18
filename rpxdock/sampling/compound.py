import numpy as np, rpxdock as sd

class CompoundHier:
   def __init__(self, *args):
      self.parts = args
      self.ncells = np.array([p.ncell for p in self.parts], dtype='u8')
      self.ncell_div = np.cumprod([1] + [p.ncell for p in self.parts[:-1]], dtype='u8')
      self.ncell = np.prod(self.ncells, dtype='u8')
      self.dims = np.array([p.dim for p in self.parts], dtype='u8')
      self.dim = np.sum(self.dims, dtype='u8')
      self.dummies = {
         np.float32: sd.sampling.DummyHier_f4(self.dim, self.ncell),
         np.float64: sd.sampling.DummyHier_f8(self.dim, self.ncell)
      }

   def check_indices(self, resl, idx):
      idx = np.array(idx, dtype='u8')
      if np.any(idx > self.size(resl)):
         raise ValueError('index too large')
      return idx

   def cell_index_of(self, resl, idx):
      idx = self.check_indices(resl, idx)
      return np.right_shift(idx, np.uint64(self.dim * resl))

   def hier_index_of(self, resl, idx):
      idx = self.check_indices(resl, idx)
      mask = np.uint64(2**(self.dim * resl) - 1)
      return np.bitwise_and(mask, idx)

   def split_indices_cell(self, resl, idx):
      idx = self.check_indices(resl, idx)
      cellidx = self.cell_index_of(resl, idx)
      cellidx = cellidx // self.ncell_div[:, None]
      return cellidx % self.ncells[:, None]

   def split_indices_hier(self, resl, idx):
      idx = self.check_indices(resl, idx)
      hieridx = self.hier_index_of(resl, idx)
      zod = sd.sampling.unpack_zorder(self.dim, resl, hieridx)
      split = list()
      for d in self.dims:
         zidx, zod = zod[:d], zod[d:]
         split.append(sd.sampling.pack_zorder(resl, zidx))
      return np.stack(split)

   def split_indices(self, resl, idx):
      idx = self.check_indices(resl, idx)
      cell = self.split_indices_cell(resl, idx)
      hier = self.split_indices_hier(resl, idx)
      assert len(cell) == len(hier)
      split = list()
      for c, h, d in zip(cell, hier, self.dims):
         c = np.left_shift(c, np.uint64(resl * d))
         split.append(np.bitwise_or(c, h))
      return np.stack(split)

   def get_xforms(self, resl, idx):
      split = self.split_indices(resl, idx)
      ok = np.repeat(True, len(idx))
      xforms = list()
      for p, sidx in zip(self.parts, split):
         v, x = p.get_xforms(resl, sidx)
         ok &= v
         xforms.append(x)
      return ok, np.stack(xforms)[:, ok]

   def expand_top_N(self, nexpand, resl, scores, indices):
      dummy = dummies[scores.dtype]
      idx, _ = dummy.expand_top_N(nexpand, resl, scores, indices)
      ok, xforms = self.get_xforms(resl + 1, idx)
      return idx[ok]

   def size(self, resl):
      return np.uint64(self.ncell * self.cellsize(resl))

   def cellsize(self, resl):
      return np.uint64(2**(self.dim * resl))
