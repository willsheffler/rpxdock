import numpy as np
import rpxdock.homog as hm
from rpxdock.geom import sym

def test_sym():
   assert 33 in sym.tetrahedral_axes
   assert 7 not in sym.tetrahedral_axes
   for ax in sym.axes.values():
      for a in ax.values():
         assert np.allclose(1, np.linalg.norm(a))

   print("neighboring component stuff")
   for arch in "TOI":
      assert np.allclose(sym.frames[arch][0], np.eye(4))
      for ax in sym.axes[arch]:
         a = sym.axes[arch][ax]
         dot = hm.hdot(a, sym.frames[arch] @ a)
         mx = np.max(dot[dot < 0.9])
         w = np.where(np.abs(dot - mx) < 0.01)[0]
         print(arch, ax, w[0], mx)
         x = sym.to_neighbor_olig[arch][ax]
         assert np.allclose(hm.hdot(a, x @ a), mx)

def test_symframes():
   assert len(sym.symframes("Tarst")) == 12
   assert len(sym.symframes("Oarst")) == 24
   assert len(sym.symframes("Iarst")) == 60
   assert len(sym.symframes("C4")) == 4

# for ide, bypass pytest
if __name__ == "__main__":
   test_sym()
   test_symframes()
