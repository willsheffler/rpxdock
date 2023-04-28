import random, os, sys
from opt_einsum import contract as einsum
from typing import List
import numpy as np
import willutil as wu
import rpxdock as rp

def main():
   # np.random.seed(1)
   kw = rp.options.get_cli_args()
   kw.dont_use_rosetta = True
   kw.ignored_aas = 'CP'
   kw.wts.ncontact = 0.01
   kw.score_only_ss = 'H'
   kw.exclude_residue_neighbors = 3
   kw.caclashdis = 5
   kw.framedistcut = 50
   kw.mc_nruns = 1
   # kw.mc_dump_initial_samples = True

   sym, *psyms = kw.architecture.upper().split('_')
   symelems = [wu.sym.symelems(sym, psym)[0] for psym in psyms]
   ic(sym)
   ic(psyms)
   ic(symelems)

   mcsym = McSymmetry(sym, symelems)
   for fnames in zip(*kw.inputs):
      components = component_from_pdb(fnames, symelems, **kw)
      search = RpxMonteCarlo(components, mcsym, **kw)
      result = search.run(**kw)

class RpxMonteCarlo:
   """manages rpx based monte-carlo protocol"""
   def __init__(
      self,
      components: 'List[McComponent]',
      mcsym: 'McSymmetry',
      caclashdis=4,
      **kw,
   ):
      self.kw = wu.Bunch(kw)
      self.components = components
      self.ncomp = len(components)
      self.compcoms = np.stack([c.com for c in self.components])
      self.mcsym = mcsym
      self.hscore = rp.RpxHier(self.kw.hscore_files, **kw)
      self.caclashdis = caclashdis
      self.results = list()
      self.reset()

   def reset(self):
      self.mcsym.reset()

   def run(
      self,
      mc_nruns,
      **kw,
   ):
      kw = self.kw.sub(kw)
      results = list()
      for isamp in range(mc_nruns):
         self.reset()
         try:
            score, samp = self.runone(isamp, **kw)
            if score >= 0: continue
            results.append((score, samp))
            self.mcsym.to_canonical_asu_position(samp, self.compcoms, forceupdate=True, **kw)
            self.dumppdbs(f'mc__{self.label()}__samp{isamp:04}', samp)

         except Exception as e:
            print('error on', isamp, flush=True)
            print(e, flush=True)
            # if isinstance(e, AssertionError):
            raise e

   def runone(
         self,
         isamp,
         mc_ntrials=1000,
         startspread=wu.Bunch(cartsd=4.0, rotsd=0.15, latticesd=8.0),
         **kw,
   ):
      kw = self.kw.sub(kw)
      # print('-' * 80, flush=True)
      spread = startspread.copy()
      sample = self.startsample()

      if kw.mc_dump_initial_samples:
         self.dumppdbs(f'start_{isamp}.pdb', sample)
         return 9e9, None

      mc = wu.MonteCarlo(self.objectivefunc, temperature=2, **kw)
      for itrial in range(mc_ntrials):
         assert sample
         newsample = self.new_sample(sample, spread)

         accept = mc.try_this(newsample)
         if accept:
            sample = newsample
            if mc.new_best_last:
               self.results.append((mc.best, mc.beststate))
         # if itrial % 100 == 0:
         # ic(itrial, spread)
         self.adjust_spread(spread, itrial)
      ic(isamp, mc.best)
      newsc = self.objectivefunc(mc.beststate)
      if not np.allclose(newsc, mc.best):
         print('rescore mismatch', isamp, newsc, mc.best)
      return mc.best, mc.beststate.copy()

   # reset to closest frame to ref hpoint

   # scale with lattice change
   # replace position/lattice w/ samp

   def startsample(self, **kw):
      kw = self.kw.sub(kw)
      position = np.stack([c.random_unitcell_position()[0] for c in self.components])
      lattice = np.eye(3)
      lattice *= 70 + np.random.rand() * 70
      lattice *= np.sqrt(len(self.components))
      position = wu.sym.applylattice(lattice, position)
      sample = wu.Bunch(position=position, lattice=lattice)
      sample = self.mcsym.to_canonical_asu_position(sample, self.compcoms, forceupdate=True, **kw)
      return sample

   def adjust_spread(self, spread, itrial):
      for k in spread:
         spread[k] *= 0.998
      # spread.cartsd = 1.0
      # spread.rotsd = 0.03
      # spread.latticesd = 2.0

   def label(self):
      return str.join('__', [c.label for c in self.components])

   def new_sample(self, sample, spread, **kw):
      kw = self.kw.sub(kw)
      position, lattice = sample.values()
      assert position.ndim == 3
      assert lattice.shape == (3, 3)
      newpos, newlat = self.mcsym.perturb_lattice(position, lattice, spread, **kw)
      pospert = np.stack([c.perturbation(spread, **kw) for c in self.components])
      newpos = newpos @ pospert
      newsample = wu.Bunch(position=newpos, lattice=newlat)
      newsample = self.mcsym.to_canonical_asu_position(newsample, self.compcoms, **kw)
      return newsample

   def objectivefunc(self, sample, **kw):
      position, lattice = sample.values()
      assert position.shape == (self.ncomp, 4, 4)
      allscores = self.score(sample, **kw)
      totcompscore = allscores.sum(axis=0)
      # totcompscore[1, 0] *= 2
      return np.sum(totcompscore)
      # fill lower tri
      # for i in range(len(totcompscore)):
      # for j in range(i):
      # totcompscore[j, i] = totcompscore[i, j]
      # return np.sort(totcompscore.flat)[1]

   def dumppdbs(self, fname, samples, dumpasym=True, dumpsym=True, **kw):
      positions, lattices = samples.values()
      positions = positions.reshape(-1, self.ncomp, 4, 4)
      lattices = lattices.reshape(-1, 3, 3)
      for ipos, (position, lattice) in enumerate(zip(positions, lattices)):
         frames = wu.sym.applylattice(lattice, self.mcsym.closeframes)
         for icomp, (pos, comp) in enumerate(zip(position, self.components)):
            cryst1 = wu.sym.cryst1_line(self.mcsym.sym, lattice)
            coords = wu.hxformpts(pos, comp.body.coord)
            if dumpasym:
               wu.dumppdb(f'{fname}_pos{ipos:04}_comp{icomp}_asym.pdb', coords, header=cryst1)
            if dumpsym:
               coords = wu.hxformpts(frames, coords)
               wu.dumppdb(f'{fname}_pos{ipos:04}_comp{icomp}_sym.pdb', coords)
            # wu.showme(coords[:100, :3], is_points=True)

   def sympositions(self, position, lattice, icomp1, icomp2):
      frames0, fmask = self.mcsym.scoreframes_for_component(icomp1, icomp2)
      nframes = len(frames0)
      frames = wu.sym.applylattice(lattice, frames0)
      pos1 = position[icomp1]
      pos2 = wu.hxformx(frames, position[icomp2])
      # pos1 = np.repeat(positions[:, icomp1], nframes, axis=0).reshape(npos, nframes, 4, 4)
      # pos2 = np.repeat(positions[:, icomp2], nframes, axis=0).reshape(npos, nframes, 4, 4)
      # pos2 = wu.hxformx(frames, pos2)
      return frames, fmask, pos1, pos2

   def score(self, samples, **kw):
      kw = self.kw.sub(kw)
      samples = self._check_samples(samples)
      positions, lattices = samples.values()

      component_combos = list()
      for icomp1, comp1 in enumerate(self.components):
         for icomp2, comp2 in enumerate(self.components):
            if icomp2 > icomp1: continue  # upper triangle
            component_combos.append(
               (icomp1, comp1, icomp2, comp2, *self.sympositions(positions, lattices, icomp1, icomp2)))

      scores = np.zeros((self.mcsym.nframes, self.ncomp, self.ncomp))
      for icomp1, comp1, icomp2, comp2, frames, fmask, pos1, pos2 in component_combos:
         compclash = comp1.body.contact_count_ca(comp2.body, pos1, pos2, self.caclashdis)
         scores[fmask, icomp1, icomp2] += 100.0 * compclash
      if scores.sum() > 0: return scores  # have clashes

      for icomp1, comp1, icomp2, comp2, frames, fmask, pos1, pos2 in component_combos:
         compscore = self.hscore.scorepos(comp1.body, comp2.body, pos1, pos2, iresl=0, **kw)
         scores[fmask, icomp1, icomp2] -= compscore
      if np.any(scores < 0): return scores

      comdist = 9e9 * np.ones((self.ncomp, self.ncomp))
      for icomp1, comp1, icomp2, comp2, frames, fmask, pos1, pos2 in component_combos:
         # don't pull intra contacts together if multicomp
         if len(self.components) > 1 and icomp1 == icomp2:
            continue
         com = wu.hxform(pos1, comp1.body.com())
         # ic(pos2.shape, comp2.com)
         symcom = wu.hxformpts(pos2, comp2.com)
         comdist[icomp1, icomp2] = np.min(wu.hnorm(com - symcom))
         scores[0, 0, 0] = 100_000 + np.min(wu.hnorm(com - symcom))
      return scores

   def _check_samples(self, sample):
      position, lattice = sample.values()
      assert position.ndim == 3 and lattice.shape == (3, 3)
      return sample
      # assert position.shape[-3:] == (self.ncomp, 4, 4)
      # assert lattice.shape[-2:] == (3, 3)
      # position = position.reshape(-1, self.ncomp, 4, 4)
      # lattice = lattice.reshape(-1, 3, 3)
      # assert len(position) == len(lattice)
      # return wu.Bunch(position=position, lattice=lattice)

class McComponent:
   """Body coupled to sym element"""
   def __init__(
      self,
      coords: np.ndarray,
      symelem: 'SymElem',
      pdbname: str,
      label: str = None,
      **kw,
   ):
      self.kw = wu.Bunch(kw)
      self.symelem = symelem
      assert coords.shape[0] == self.symelem.numops
      self._init_coords = coords.copy()
      self._aligned_coords = wu.sym.align(coords, self.symelem)
      self.body = rp.Body(self._aligned_coords[0], **kw)
      self.com = self.body.com()
      self.pdbname = pdbname
      self.label = label or str.join(',', os.path.basename(self.pdbname).split('.')[:-1])

   def random_unitcell_position(self, size=1):
      '''random rotation and placement in unit cell'''
      se = self.symelem
      if se.iscyclic:
         cen = np.random.rand(size) * 2000 - 1000  # way big to avoid boundary effects
         cen = (se.axis[None] * cen[:, None] + se.cen) % 1.0
         offset = wu.htrans(cen)

         flipangle = np.where(np.random.rand(size) < 0.5, 0, np.pi)

         perp = wu.hnormalized(wu.hcross(se.axis, [1, 2, 3]))
         randrot = wu.hrot(se.axis, np.random.rand(size) * 2 * np.pi, cen)
         fliprot = wu.hrot(perp, flipangle, cen)
         pos = wu.hxformx(wu.hxformx(randrot, fliprot), offset)
         return pos
      elif se.isdihedral and se.nfold > 2:
         assert 0
         alignangle = np.where(np.random.rand(size) < 0.5, 0, np.pi / se.nfold)
         return wu.hrot(se.axis, alignangle, cen)
      elif se.isdihedral and se.nfold == 2:
         a1, a2, a3 = se.axis, se.axis2, wu.hcross(se.axis, se.axis2)
         oris = np.stack([
            wu.halign2(a1, a2, a1, a2),
            wu.halign2(a1, a2, a1, a3),
            wu.halign2(a1, a2, a2, a1),
            wu.halign2(a1, a2, a2, a3),
            wu.halign2(a1, a2, a3, a1),
            wu.halign2(a1, a2, a3, a2),
         ])
         return oris[np.randint(0, 6, size)]
      else:
         assert 0

   def perturbation(self, spread, size=1, **kw):
      '''rotation around origin, should be applied first'''
      assert size == 1
      if self.symelem.isdihedral:
         return np.tile(np.eye(4), (size, 1, 1))
      rot = np.random.normal(0, spread.rotsd, size)
      x = wu.hrot(self.symelem.axis, rot)
      cart = np.random.normal(0, spread.cartsd, size)
      x[..., 3] += cart * self.symelem.axis
      return x[0]

class McSymmetry:
   """manage symmetry and component neighbor relationships"""
   def __init__(
      self,
      sym: str,
      symelems: 'List[SymElem]',
      **kw,
   ):
      self.kw = wu.Bunch(kw)
      self.sym = sym
      self.symelems = symelems
      # self.unitasucen = wu.hpoint(asucen)
      self.allframes = wu.sym.frames(self.sym, sgonly=True, cells=5)
      # self.setup_component_frames()
      self.reset()

   def reset(self):
      self.celloffset = np.array([0.0, 0.0, 0.0, 0.0])
      self.last_closeframes_cen = np.array([9e9, 9e9, 9e9, 1])

   def perturb_lattice(self, position, lattice, spread, **kw):
      assert position.ndim == 3
      assert lattice.shape == (3, 3)
      ltype = wu.sym.latticetype(self.sym)
      if ltype == 'CUBIC':
         unitpos = wu.sym.tounitframes(position, lattice)
         newlattice = lattice + np.eye(3) * np.random.normal(0, spread.latticesd, 1)
         newpos = wu.sym.applylattice(newlattice, unitpos)
         return newpos, newlattice
      else:
         raise NotImplementedError(f'cant do lattice type {ltype}')

   def setup_component_frames(self, **kw):
      kw = self.kw.sub(kw)
      # self.closeframes = wu.sym.frames(self.sym, sgonly=True, cells=3, asucen=asucen, **kw)
      self.nframes = len(self.closeframes)
      ncomp = len(self.symelems)
      self.scoreframes = np.ones((ncomp, ncomp, self.nframes), dtype=bool)
      self.frames_by_component = list()
      testframes = self.closeframes
      for icomp, symelem in enumerate(self.symelems):
         self.frames_by_component.append(list())
         ops = wu.htrans(-self.celloffset) @ symelem.operators @ wu.htrans(self.celloffset)
         # debug2 = debug - self.celloffset
         # wu.dumppdb(f'/home/sheffler/project/rpxtal/canon.pdb', debug, frames=np.eye(4))
         # wu.dumppdb(f'/home/sheffler/project/rpxtal/canonoffset.pdb', debug2, frames=np.eye(4))
         # wu.dumppdb(f'/home/sheffler/project/rpxtal/ref.pdb', debug2, frames=ops)
         # wu.dumppdb(f'/home/sheffler/project/rpxtal/all.pdb', debug2, frames=self.closeframes)
         # ic(self.celloffset)
         # # ic(wu.haxis_ang_cen_of(ops))
         # ic(wu.haxis_ang_cen_of(ops)[2][1:, :3])
         for iframe, frame in enumerate(testframes):
            # if np.allclose(frame[:3, :3], ops[1, :3, :3]):
            # ic(frame)
            # wu.dumppdb(f'/home/sheffler/project/rpxtal/test_{iframe}.pdb', debug2, frames=frame)
            if np.any(np.all(np.isclose(frame, ops, atol=1e-6), axis=(1, 2))):
               self.scoreframes[icomp, icomp, iframe] = False
         for icomp2, symelem2 in enumerate(self.symelems):
            frames_c1c2 = self.closeframes[self.scoreframes[icomp, icomp2]]
            self.frames_by_component[icomp].append(frames_c1c2)

         # ic(icomp, symelem)
         # ic(np.sum(~self.scoreframes[icomp, icomp]))
         assert np.sum(~self.scoreframes[icomp, icomp]) == symelem.numops

      norot = np.all(np.isclose(self.closeframes[:, :3, :3], np.eye(3)), axis=(1, 2))
      # self.transframes = self.closeframes[norot]

   def scoreframes_for_component(self, icomp1, icomp2):
      return (
         self.frames_by_component[icomp1][icomp2],
         self.scoreframes[icomp1, icomp2],
      )

   def frames(self, lattice, unitframes=None):
      if unitframes is None: unitframes = self.closeframes
      nframes = len(unitframes)
      # frames = np.tile(unitframes, (npos, 1, 1)).reshape(npos, nframes, 4, 4)
      frames = wu.sym.applylattice(lattice, unitframes)
      return frames

   def update_closeframes(self, position, lattice, com, framedistcut, forceupdate=False, **kw):
      poscom = einsum('cij,cj->ci', position, com)
      # ic(poscom)
      paddingfrac = 3
      dist = [wu.hnorm(poscom[i] - self.last_closeframes_cen[i]) for i in range(len(poscom))]
      close = [d < framedistcut / paddingfrac for d in dist]
      # ic(close)
      if all(close) and not forceupdate:
         return
      # ic('update')
      self.last_closeframes_cen = poscom
      # self.last_closeframes_lattice = np.sum(lattice.diagonal(), axis=0)
      frames = self.frames(lattice, self.allframes)
      close = np.zeros(len(frames), dtype=bool)
      for i, cen in enumerate(poscom):
         symcom = einsum('fij,j->fi', frames, cen)
         dist = wu.hnorm(symcom - cen)
         close |= dist < framedistcut * (1 + 1 / paddingfrac)
      self.closeframes = self.allframes[close]
      # self.closeframes = self.allframes[np.argsort(dist2)[:nclose]]

      # compcom = einsum('cij,cj->ci', position, com)
      # ic(compcom)
      # wu.dumppdb('compcom.pdb', compcom)
      # wu.dumppdb('poscom.pdb', poscom[None])
      # frames = self.frames(lattice, self.closeframes)
      # wu.dumppdb('closecom.pdb', poscom[None], frames=frames)
      # wu.dumppdb('symcom.pdb', symcom)
      # assert 0

      # self.closeframes = self.allframes[dist2 < comdistcut**2]
      self.setup_component_frames(**kw)

   def to_canonical_asu_position(self, sample, com, forceupdate=False, **kw):
      position, lattice = sample.values()

      # position = position.copy()
      # assert position.ndim == 3 and lattice.shape == (3, 3) and com.ndim == 2
      # poscom = einsum('cij,cj->i', position, com) / len(com)
      # unitcom = np.linalg.inv(lattice) @ poscom[:3]
      # delta = unitcom - unitcom % 1.0
      # if np.allclose(delta, 0) and not forceupdate:
      #    return sample
      # self.celloffset[:3] += delta
      # position[:, :3, 3] -= delta * lattice.diagonal()
      # ic(self.celloffset, delta)
      # ic(position[:, :3, 3])

      self.update_closeframes(position, lattice, com, forceupdate=forceupdate, **kw)
      newsample = wu.Bunch(position=position, lattice=lattice)
      # print(repr(newsample), flush=True)
      return newsample

      # # origshape = position.shape
      # candidateframes = self.transframes
      # com = com.reshape(-1, 4)
      # # position = position.reshape(-1, len(com), 4, 4)
      # poscom = einsum('cij,cj->ci', position, com)
      # symcom = wu.hxformpts(candidateframes, poscom)
      # # asucen = wu.hpoint(lattice @ self.unitasucen[:3])
      # # ic(asucen)
      # dist = wu.hnorm(symcom)
      # ic(dist)
      # w = np.argmin(dist, axis=0)
      # ic(w.shape, w, dist[w])
      # ic(candidateframes[w].shape)
      # newpos = einsum('cij,cjk->cik', candidateframes[w], position)
      # # assert 0
      # ic(newpos.shape, com.shape)
      # newcom = einsum('cij,cj->ci', newpos, com)
      # asucen = einsum('ij,cj->ci', np.linalg.inv(lattice), newcom[:, :3])
      # ic(asucen.shape, asucen)
      # self.setup_component_frames(asucen=asucen)
      # return wu.Bunch(position=newpos, lattice=lattice)

def component_from_pdb(fname, symelem, **kw):
   if isinstance(fname, str):
      pdb = wu.readpdb(fname)
      coords = pdb.ncaco(splitchains=True)
      return McComponent(coords, symelem, pdbname=fname, **kw)
      # coords = wu.sym.align(coords, sym='Cx', symelem=symelem)
      # b = rp.Body(coords[0], **kw)
      # return McComponent(b, symelem)
   return [component_from_pdb(f, s, **kw) for (f, s) in zip(fname, symelem)]

   # sample positions

#
# score positions
#
# mc check

# xform, cell

if __name__ == '__main__':
   main()
