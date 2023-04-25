import random
from typing import List
import numpy as np
import willutil as wu
import rpxdock as rp

# def perturbation(cartsd=1, rotsd=0.05, axes=None, size=1):
#    perturb = np.tile(np.eye(4), size)
#    if axes is None:
#       assert 0

#    for axs, cen in axes:
#       ang = np.random.normal(size=size) * rotsd
#       crt = np.random.normal(size=size) * cartsd * axs
#       x = wu.hrot(axs, ang, cen)
#       x[..., :3, 3] = crt[..., :3]
#       perturb = wu.hxform(x, perturb)

#    return perturb

class McComponent:
   """Body coupled to sym element"""
   def __init__(
      self,
      coords: np.ndarray,
      symelem: 'SymElem',
      **kw,
   ):
      self.kw = wu.Bunch(kw)
      self.symelem = symelem
      assert coords.shape[0] == self.symelem.numops
      self._init_coords = coords.copy()
      self._aligned_coords = wu.sym.align(coords, self.symelem)
      self.body = rp.Body(self._aligned_coords[0], **kw)

   def start_position(self, size=1):
      se = self.symelem
      if se.iscyclic:
         flipangle = np.where(np.random.rand(size) < 0.5, 0, np.pi)
         perp = wu.hnormalized(wu.hcross(se.axis, [1, 2, 3]))
         return wu.hrot(perp, flipangle, se.cen)
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

   def perturbation(self, size=1, cartsd=1, rotsd=0.03):
      if self.symelem.isdihedral:
         return np.tile(np.eye(4), (size, 1, 1))
      rot = np.random.normal(0, rotsd, size)
      x = wu.hrot(self.symelem.axis, rot, self.symelem.cen)
      cart = np.random.normal(0, cartsd, size)
      x[..., 3] += cart * self.symelem.axis
      return x

class McSymmetry:
   """manage symmetry and component neighbor relationships"""
   def __init__(
      self,
      sym: str,
      symelems: 'List[SymElem]',
      asucen=[0.5, 0.5, 0.5],
      **kw,
   ):
      self.kw = wu.Bunch(kw)
      self.sym = sym
      self.symelems = symelems
      self.asucen = wu.hpoint(asucen)
      self.setup_frames()

   def perturb_lattice(self, position, lattice, cartsd=2, **kw):
      assert lattice.shape == (3, 3)
      ltype = wu.sym.latticetype(self.sym)
      if ltype == 'CUBIC':
         unitpos = wu.sym.tounitframes(position, lattice)
         newlattice = lattice + np.eye(3) * np.random.normal(0, cartsd, 1)
         newpos = wu.sym.applylattice(unitpos, newlattice)
         return newpos, newlattice
      else:
         raise NotImplementedError(f'cant do lattice type {ltype}')

   def setup_frames(self, **kw):
      kw = self.kw.sub(kw)
      self.allframes = wu.sym.frames(self.sym, sgonly=True, cells=3, asucen=self.asucen, **kw)
      self.nframes = len(self.allframes)
      ncomp = len(self.symelems)
      self.scoreframes = np.ones((ncomp, ncomp, self.nframes), dtype=bool)
      self.frames_by_component = list()
      for icomp, symelem in enumerate(self.symelems):
         self.frames_by_component.append(list())
         for iframe, frame in enumerate(self.allframes):
            if np.any(np.all(np.isclose(frame, symelem.operators), axis=(1, 2))):
               self.scoreframes[icomp, icomp, iframe] = False
         for icomp2, symelem2 in enumerate(self.symelems):
            frames_c1c2 = self.allframes[self.scoreframes[icomp, icomp2]]
            self.frames_by_component[icomp].append(frames_c1c2)
         assert np.sum(~self.scoreframes[icomp, icomp]) == symelem.numops

   def scoreframes_for_component(self, icomp1, icomp2):
      return (
         self.frames_by_component[icomp1][icomp2],
         self.scoreframes[icomp1, icomp2],
      )

class RpxMonteCarlo:
   """manages rpx based monte-carlo protocol"""
   def __init__(
      self,
      components: 'List[McComponent]',
      mcsym: 'McSymmetry',
      **kw,
   ):
      self.kw = wu.Bunch(kw)
      self.components = components
      self.ncomp = len(components)
      self.mcsym = mcsym
      self.hscore = rp.RpxHier(self.kw.hscore_files, **kw)

   def run(self, **kw):
      kw = self.kw.sub(kw)

      nsamp = 1

      for isamp in range(10):

         positions = np.tile(np.eye(4), (nsamp, self.ncomp)).reshape(nsamp, self.ncomp, 4, 4)
         lattices = np.tile(np.eye(3), (nsamp, 1, 1)).reshape(nsamp, 3, 3)
         positions[:, 0, :3, 3] = 60 + np.random.normal(0, 7, nsamp)

         lattices *= 150 + np.random.normal(0, 20, nsamp)
         sample = wu.Bunch(position=positions[0], lattice=lattices[0])

         mc = wu.MonteCarlo(self.objectivefunc, temperature=2, **kw)
         for imc in range(1000):
            newsample = self.new_sample(sample)
            accept = mc.try_this(newsample)
            if accept:
               sample = newsample
               # if mc.new_best_last:
               # ic('best', imc, mc.low)
            if imc % 1000 == 0:
               sample = mc.beststate
               mc.temperature *= 0.8
         ic('best', mc.best)
         self.dumppdbs(f'samp_{isamp}.pdb', mc.beststate)

   # reset to closest frame to ref hpoint

   # scale with lattice change
   # replace position/lattice w/ samp

   def new_sample(self, sample, **kw):
      position, lattice = sample.values()
      nsamp = len(position)
      newpos, newlat = self.mcsym.perturb_lattice(position, lattice, **kw)
      pospert = np.stack([c.perturbation(size=nsamp, **kw) for c in self.components])
      pospert = pospert.swapaxes(0, 1)
      newpos = wu.hxformx(pospert, position, outerprod=False).reshape(position.shape)
      return wu.Bunch(position=newpos, lattice=newlat)

   def objectivefunc(self, sample, **kw):
      position, lattice = sample.values()
      assert position.shape == (self.ncomp, 4, 4)
      allscores = self.score(sample, **kw)
      return allscores.sum()

   def dumppdbs(self, fname, samples, **kw):
      positions, lattices = samples.values()
      positions = positions.reshape(-1, self.ncomp, 4, 4)
      lattices = lattices.reshape(-1, 3, 3)
      for ipos, (position, lattice) in enumerate(zip(positions, lattices)):
         frames = wu.sym.applylattice(self.mcsym.allframes, lattice)
         for icomp, (pos, comp) in enumerate(zip(position, self.components)):
            coords = wu.hxformpts(pos, comp.body.coord)
            coords = wu.hxformpts(frames, coords)
            wu.dumppdb(f'{fname}_pos{ipos:04}_comp{icomp}.pdb', coords)
            # wu.showme(coords[:100, :3], is_points=True)

   def sympositions(self, positions, lattices, icomp1, icomp2):
      npos = len(positions)
      frames0, fmask = self.mcsym.scoreframes_for_component(icomp1, icomp2)
      nframes = len(frames0)
      # ic(nframes, np.sum(fmask))
      frames = np.tile(frames0, (npos, 1, 1)).reshape(npos, len(frames0), 4, 4)
      frames = wu.sym.applylattice(frames, lattices)
      pos1 = np.repeat(positions[:, icomp1], nframes, axis=0).reshape(npos, nframes, 4, 4)
      pos2 = np.repeat(positions[:, icomp2], nframes, axis=0).reshape(npos, nframes, 4, 4)
      pos2 = wu.hxformx(frames, pos2)
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

      clash = np.zeros(len(positions))
      for icomp1, comp1, icomp2, comp2, frames, fmask, pos1, pos2 in component_combos:
         compclash = comp1.body.contact_count(comp2.body, pos1, pos2, maxdis=4.0)
         clash += compclash.sum(axis=1)

      scores = np.zeros((len(positions), self.mcsym.nframes, self.ncomp, self.ncomp))
      if np.all(clash): return scores
      ok = (clash == 0)
      for icomp1, comp1, icomp2, comp2, frames, fmask, pos1, pos2 in component_combos:
         compscore = self.hscore.scorepos(comp1.body, comp2.body, pos1[ok], pos2[ok], iresl=0, **kw)
         for i, w in enumerate(np.where(ok)[0]):
            scores[w, fmask, icomp1, icomp2] -= compscore[i]

      missing = np.all(scores == 0, axis=(1, 2, 3))
      if not np.any(missing): return scores
      comdist = 9e9 * np.ones((len(positions), self.ncomp, self.ncomp))
      for icomp1, comp1, icomp2, comp2, frames, fmask, pos1, pos2 in component_combos:
         for i, w in enumerate(np.where(missing)[0]):
            com = wu.hxform(pos1[i], comp1.body.com())
            symcom = wu.hxform(pos2[i], comp2.body.com())
            comdist[w, icomp1, icomp2] = np.min(wu.hnorm(com - symcom))
            scores[w, 0, 0, 0] = 100_000 + np.min(wu.hnorm(com - symcom))

      return scores

   def _check_samples(self, samples):
      positions, lattices = samples.values()
      assert positions.shape[-3:] == (self.ncomp, 4, 4)
      assert lattices.shape[-2:] == (3, 3)
      positions = positions.reshape(-1, self.ncomp, 4, 4)
      lattices = lattices.reshape(-1, 3, 3)
      assert len(positions) == len(lattices)
      return wu.Bunch(position=positions, lattice=lattices)

def main():
   kw = rp.options.get_cli_args()
   kw.dont_use_rosetta = True
   kw.ignored_aas = 'CP'
   sym = 'I 21 3'
   symelems = [wu.sym.SymElem(3, [1, 1, 1])]
   mcsym = McSymmetry(sym, symelems, xtalrad=0.5)

   for fnames in kw.inputs:
      components = component_from_pdb(fnames, symelems, **kw)
      search = RpxMonteCarlo(components, mcsym, **kw)
      result = search.run()

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
