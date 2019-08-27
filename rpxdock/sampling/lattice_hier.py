class LatticeHier:
   def __init__(self, components, directions, lb, ub, nstep, ncell=1):
      assert len(components) == len(directions) - 1
      self.hier1d = rp.sampling.CartHier1D_f4([lb], [ub], [nstep])

   def get_xforms(self, self, resl=0, idx):
      if idx is None: idx = np.arange(self.size(resl))
      split = self.split_indices(resl, idx)
      ok = np.repeat(True, len(idx))
      xforms = list()
      for p, sidx in zip(self.parts, split):
         v, x = p.get_xforms(resl, sidx)
         ok &= v
         xforms.append(x)
      return ok, np.stack(xforms, axis=1)[ok]

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
