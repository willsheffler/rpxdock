from rpxdock.sampling import CompoundHier
import numpy as np

class LatticeHier(CompoundHier):
   def __init__(self, parts, directions):
      super().__init__(*parts)
      if len(directions) + 1 != len(parts):
         raise ArgumentError('must be one direction for each part, other than the first')
      self.directions = directions
      for d in directions:
         assert d[2] == 0

   def combine_xforms(self, xparts):
      offset = xparts[:, 0, 2, 3]
      for i, d in enumerate(self.directions):
         xparts[:, i + 1, 0, 3] = offset * d[0]
         xparts[:, i + 1, 1, 3] = offset * d[1]
      xparts[:, 0, 2, 3] = 0  #
      return xparts

   def get_xforms(self, *args, **kw):
      ok, xparts = super().get_xforms(*args, **kw)
      return ok, self.combine_xforms(xparts)

   def expand_top_N(self, nexpand, resl, scores, indices):
      idx, xparts = super().expand_top_N(nexpand, resl, scores, indices)
      return idx, self.combine_xforms(xparts)
