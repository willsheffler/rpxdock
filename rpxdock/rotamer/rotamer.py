import numpy as np
from rpxdock.rotamer.richardson import get_rotamer_space

def assign_rotamers(rp, rotspace=None):
   if rotspace is None:
      rotspace = get_rotamer_space()

   rotlbl = list("GA")
   rotids = -np.ones(len(rp.ca), dtype="i4")
   rotids[rp.aaid == rp.aa2id.sel(aa="G")] = 0
   rotids[rp.aaid == rp.aa2id.sel(aa="A")] = 1
   allrotchi = [np.array([])] * 2
   nrot = 2
   for aa, nchi in aa_nchi.items():
      if nchi == 0: continue
      aaid = int(rp.aa2id.sel(aa=aa))
      naa = np.sum(rp.aaid == aaid)
      if naa == 0: continue
      rs = rotspace.sel(aa=aa)
      rotchi = np.stack([rs["x" + str(i + 1)] for i in range(nchi)], axis=1)
      assert rotchi.shape[0] == len(rs.lbl)
      allrotchi.extend(rotchi)
      chi = np.stack([rp["chi" + str(i + 1)][rp.aaid == aaid] for i in range(nchi)], axis=1)[:,
                                                                                             None]
      assert -180.0 <= np.min(chi)
      assert np.max(chi) <= 180.0
      # print(aa, rotchi.shape, chi.shape)
      # chimul = np.array([4, 3, 2, 1])[:nchi]
      chimul = np.array([1, 1, 1, 1])[:nchi]
      diff = (chi - rotchi[None]) * chimul
      diff2 = np.minimum(diff**2, (diff - 360 * chimul)**2)
      diff2 = np.minimum(diff2, np.abs(diff + 360 * chimul)**2)
      d2 = np.sum(diff2, axis=2)
      imin = np.argmin(d2, axis=1)

      rotids[rp.aaid == aaid] = nrot + imin
      nrot += len(rs.lbl)
      rotlbl.extend(aa + l for l in rs.lbl.data)
      # print(imin.shape, np.max(imin))
      # for i in range(len(rs.lbl)):
      # print(i, np.sum(imin == i))

   for i in range(len(rotlbl)):
      aa = rotlbl[i][0]
      aaid = int(rp.aa2id.sel(aa=aa))
      assert np.all(rp.aaid[rotids == i] == aaid)
   assert len(allrotchi) == len(rotlbl)
   return rotids, rotlbl, allrotchi

def check_rotamer_deviation(rp, rotspace, quiet=False):
   rotlbl = rp.rotlbl
   means = np.full((len(rotlbl), 4), np.nan)
   sds = np.full((len(rotlbl), 4), np.nan)
   for irot in range(2, len(rotlbl)):
      aa = rotlbl[irot][0]
      aaid = int(rp.aa2id.sel(aa=aa))
      aars = rotspace["aa"][irot - 2]
      assert aa == aars
      nchi = aa_nchi[aa]
      if np.sum(rp.rotid == irot) == 0:
         continue
      rotchi = np.array([rotspace["x" + str(i + 1)][irot - 2] for i in range(nchi)])
      chi = np.stack([rp["chi" + str(i + 1)][rp.rotid == irot] for i in range(nchi)], axis=1)
      diff = np.minimum(np.abs(chi - rotchi), np.abs(chi - rotchi + 360.0))
      diff = np.minimum(diff, np.abs(chi - rotchi - 360))
      m = np.mean(diff, axis=0)
      s = np.std(diff, axis=0)
      means[irot, :nchi] = m
      sds[irot, :nchi] = s
      if not quiet:
         for i in range(nchi):
            print(f"{irot:3} {i} {rotchi[i]:6.1f} {m[i]:6.1f} {s[i]:5.1f} {rotlbl[irot]}")
   m = np.nanmean(means)
   s = np.nanstd(sds)
   print("avg mean diff", m, "avg mean sd", s)
   return m, s

aa_nchi = dict(
   A=0,
   C=1,
   D=2,
   E=3,
   F=2,
   G=0,
   H=2,
   I=2,
   K=4,
   L=2,
   M=3,
   N=2,
   P=1,
   Q=3,
   R=4,
   S=1,
   T=1,
   V=1,
   W=2,
   Y=2,
)
