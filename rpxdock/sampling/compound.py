import numpy as np, rpxdock as sd

class ZeroDHier:
   def __init__(self, samples):
      if isinstance(samples, ZeroDHier):
         self.samples = samples.samples
      else:
         samples = np.asarray(samples)
         if samples.ndim == 2: samples = samples[None]
         if not samples.shape[-2:] == (4, 4):
            raise ValueError('samples must be (N,4,4) array')
         self.samples = samples
      self.dim = 0
      self.ncell = len(self.samples)

   def size(self, resl):
      return self.ncell

   def cellsize(self, resl):
      return 1

   def __len__(self):
      return self.ncell

   def get_xforms(self, resl=0, idx=None):
      if idx is None: return self.samples
      return np.repeat(True, len(idx)), self.samples[idx]

class CompoundHier:
   def __init__(self, *args):
      self.parts = args
      self.ncells = np.array([p.ncell for p in self.parts], dtype='u8')
      self.ncell_div = np.cumprod([1] + [p.ncell for p in self.parts[:-1]], dtype='u8')
      self.ncell = np.prod(self.ncells, dtype='u8')
      self.dims = np.array([p.dim for p in self.parts], dtype='u8')
      self.dim = np.sum(self.dims, dtype='u8')
      self.dummies = {
         np.dtype('f4'): sd.sampling.DummyHier_f4(self.dim, self.ncell),
         np.dtype('f8'): sd.sampling.DummyHier_f8(self.dim, self.ncell)
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

   def get_xforms(self, resl=0, idx=None):
      if idx is None: idx = np.arange(self.size(resl))
      split = self.split_indices(resl, idx)
      ok = np.repeat(True, len(idx))
      xforms = list()
      for p, sidx in zip(self.parts, split):
         v, x = p.get_xforms(resl, sidx)
         ok &= v
         xforms.append(x)
      return ok, np.stack(xforms, axis=1)[ok]

   def expand_top_N(self, nexpand, resl, scores, indices):
      dummy = self.dummies[scores.dtype]
      idx, _ = dummy.expand_top_N(nexpand, resl, scores, indices)
      ok, xforms = CompoundHier.get_xforms(self, resl + 1, idx)
      return idx[ok], xforms

   def size(self, resl):
      return np.uint64(self.ncell * self.cellsize(resl))

   def cellsize(self, resl):
      return np.uint64(2**(self.dim * resl))

class ProductHier(CompoundHier):
   def __init__(self, *args):
      super().__init__(*args)

   def combine_xforms(self, xparts):
      x = xparts[:, -1]
      for i in reversed(range(xparts.shape[1] - 1)):
         x = xparts[:, i] @ x
      return x

   def get_xforms(self, *args, **kw):
      ok, xparts = super().get_xforms(*args, **kw)
      return ok, self.combine_xforms(xparts)

   def expand_top_N(self, nexpand, resl, scores, indices):
      idx, xparts = super().expand_top_N(nexpand, resl, scores, indices)
      return idx, self.combine_xforms(xparts)

class SlideHier:
   def __init__(self, sampler, body1, body2):
      self.sampler = sampler
      if isintance(sampler, (list, tuple)):
         assert len(sampler) is 3
         self.sampler = rp.CompoundHier(*sampler)
      self.body1 = body1
      self.body2 = body2

   def get_xforms(self, resl, idx):
      pass
