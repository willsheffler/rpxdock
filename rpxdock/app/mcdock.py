import random, os, sys, itertools as it, sys, concurrent.futures as cf
from pprint import pprint
from opt_einsum import contract as einsum
from typing import List
import numpy as np
import willutil as wu
import rpxdock as rp

# P312 missing C2 elem
# P321
# R32 2fold comp broken
# P6 bad component frames
# P63
# P6322

# P121  ok
# C121
# P222
# P2221
# P21212
# C2221
# C222
# F222
# I222
# I212121
# P4
# P42
# I4
# I41
# P422
# P4212
# P4122
# P41212
# P4222
# P42212
# P4322
# P43212
# I422
# I4122
# P321
# P3121
# P3221
# R32
# P6
# P62
# P64
# P6122
# P6522
# P6222
# P6422
# P6322

def main():

   # for k, v in wu.sym.sg_symelem_dict.items():
   # if any([e.nfold == 2 and e.iscyclic for e in v]):
   # print(k)

   kw = rp.options.get_cli_args()
   if kw.mc_random_seed is None:
      kw.mc_random_seed = np.random.randint(2**32 - 1)
   print(f'random seed: {kw.mc_random_seed}')
   np.random.seed(kw.mc_random_seed)

   # kw.use_rosetta = False
   kw.ignored_aas = 'CP'
   # kw.wts.ncontact = 0.0001
   # kw.score_only_ss = 'HEL'
   # kw.exclude_residue_neighbors = 0
   kw.wts.ncontact = 0.000
   kw.score_only_ss = 'HE'
   kw.exclude_residue_neighbors = 3
   kw.caclashdis = 5
   kw.debug = False

   # kw.mc_framedistcut = 60
   # kw.mc_intercomp_only = True
   pprint(kw.hscore_files)
   ic(kw.mc_framedistcut)
   ic(kw.mc_cell_bounds)
   ic(kw.mc_min_solvfrac, kw.mc_max_solvfrac)
   ic(kw.mc_min_contacts, kw.mc_max_contacts)
   # kw.hscore_files = 'ilv_h'
   timer = wu.Timer()

   kw.hscore = rp.RpxHier(kw.hscore_files, **kw)

   sym, *psyms = kw.architecture.upper().split('_')
   symelems = list()
   for i, psym in enumerate(psyms):
      psymelems = wu.sym.symelems(sym, psym)
      ic(kw.mc_which_symelems)
      if kw.mc_which_symelems[i] >= len(psymelems):
         raise ValueError(f'You requested element {kw.mc_which_symelems[i]}, but there are only '
                          f'{len(psymelems)} symelems for point sym {psym} in spacegroup {sym}\n'
                          f'All available symmetry elements for {sym}:\n'
                          f'{repr(wu.sym.symelems(sym))}')
      symelems.append(psymelems[kw.mc_which_symelems[i]])

   pprint(sym)
   pprint(psyms)
   pprint(wu.sym.symelems(sym))
   pprint(symelems)

   mcsym = McSymmetry(sym, symelems, **kw)
   if isinstance(kw.inputs[0], str):
      ncomp = len(psyms)
      kw.inputs = [kw.inputs[i::ncomp] for i in range(ncomp)]
   for iinput, fnames in enumerate(zip(*kw.inputs)):
      print('input', iinput, fnames, flush=True)
      components = component_from_pdb(fnames, symelems, mcsym=mcsym, **kw)
      search = RpxMonteCarlo(components, mcsym, **kw)
      result = search.run(**kw)

   timer.report()

class RpxMonteCarlo:
   """manages rpx based monte-carlo protocol"""
   def __init__(
      self,
      components: 'List[McComponent]',
      mcsym: 'McSymmetry',
      caclashdis=4,
      hscore: 'RphHier' = None,
      **kw,
   ):
      self.kw = wu.Bunch(kw)
      self.components = components
      self.ncomp = len(components)
      self.compcoms = np.stack([c.com for c in self.components])
      self.compvols = [c.guess_volume() for c in self.components]
      self.mcsym = mcsym
      self.hscore = hscore or rp.RpxHier(self.kw.hscore_files, **kw)
      self.caclashdis = caclashdis
      self.results = list()
      self.reset()
      self.kw.mc_cell_bounds = np.array(list(zip(*self.kw.mc_cell_bounds)))
      self.component_sample_counter = 0
      self.sanity_check()

   def sanity_check(self):
      kw = self.kw
      lownfold = [e.nfold for e in self.mcsym.allsymelems if e.screw == 0]
      if lownfold and min(lownfold) > self.kw.mc_max_contacts + 1:
         print('WARNING! you have set mc_max_contacts to {self.kw.mc_max_contacts} but the lowest order'
               f'sym operator is {min(lownfold)}. setting mc_max_contacts to {min(lownfold)-1} unless magic flag'
               ' --I_KNOW_WHAT_I_AM_DOING is used')
         if not kw.I_KNOW_WHAT_I_AM_DOING:
            self.kw.mc_max_contacts = min(lownfold) - 1
      assert kw.mc_max_contacts >= kw.mc_min_contacts
      assert kw.mc_max_solvfrac > kw.mc_min_solvfrac

   def reset(self):
      self.mcsym.reset()

   def run(
      self,
      mc_nruns,
      nprocess=1,
      **kw,
   ):
      kw = self.kw.sub(kw)
      for isamp in range(mc_nruns):
         global _STUPID
         _STUPID = isamp
         for ierr in range(100):
            # try:
            self.runone(isamp, **kw)
            break
         # except Exception as e:
         # print('error on', isamp, flush=True)
         # print(e, flush=True)
         # raise e
      self.dump_results(kw.output_prefix + 'scores.txt', **kw)

   def runone(
         self,
         isamp,
         mc_ntrials=1000,
         startspread=wu.Bunch(cartsd=4.0, rotsd=0.15, latticesd=8.0),
         mc_temperature=3,
         **kw,
   ):
      kw = self.kw.sub(kw)
      # print('-' * 80, flush=True)
      spread = startspread.copy()
      sample = self.startsample()

      if kw.mc_dump_initial_samples:
         # ic(sample.position[:, :3, 3])
         # assert 0
         self.dumppdbs(f'start_{isamp}', sample, dumpasym=False, rawposition=True)
         assert 0
         return 9e9, None

      # ic(wu.hvalid(sample.position))
      # ic(sample.lattice)
      # self.dumppdbs(f'start_{isamp}', sample, dumpasym=False, rawposition=True)
      # assert 0

      mc = wu.MonteCarlo(self.objectivefunc, temperature=mc_temperature, **kw)
      for itrial in range(mc_ntrials):
         assert sample
         newsample = self.new_sample(sample, spread)
         accept = mc.try_this(newsample)
         if accept:
            sample = newsample
            # ic(itrial)
            # self.dumppdbs(f'{itrial}', sample, dumpsym=False)
         self.adjust_spread(spread, itrial)

      self.record_result(isamp, mc.beststate)

   def record_result(self, isamp, sample, **kw):
      kw = self.kw.sub(kw)
      # assert wu.hvalid_norm(sample.position)
      sample = self.mcsym.to_canonical_asu_position(sample, self.compcoms, forceupdate=True, **kw)
      # assert wu.hvalid_norm(sample.position)
      framescores, comdist = self.score(sample, **kw)
      score = self.objectivefunc(sample, **kw)
      compscores = self.mcsym.component_score_summary(framescores)
      prefix = f'{kw.output_prefix}{self.label()}_{isamp:04}'
      # pdbfiles = self.dumppdbs(prefix, sample, rawposition=True, whichcomp=[(1, 1, 109)])
      if score < 100:
         pdbfiles = self.dumppdbs(prefix, sample, rawposition=False)
         r = wu.Bunch(
            score=score,
            compscores=compscores,
            solvfrac=self.guess_solvfrac(sample),
            pdbfiles=pdbfiles,
         )
         print(f'sample {isamp:4}', _output_line(r), flush=True)
         self.results.append(r)

   def dump_results(self, fname, **kw):
      kw = self.kw.sub(kw)
      with open(fname, 'w') as out:
         for r in self.results:
            line = _output_line(r)
            line += ' ' + os.path.abspath(r.pdbfiles[0])
            out.write(line + '\n')

   def startsample(self, **kw):
      kw = self.kw.sub(kw)
      position = np.stack([c.random_unitcell_position()[0] for c in self.components])
      lattice = self.mcsym.random_lattice(component_volume=sum(self.compvols), **kw)
      # ic(lattice)
      # ic(position[0, :3, :3])
      assert wu.hvalid_norm(position)
      position = wu.sym.applylattice(lattice, position)
      # ic(position[0, :3, :3])
      assert wu.hvalid_norm(position)
      # wu.showme(position, scale=10)
      # assert wu.hvalid(position)
      # assert wu.hvalid_norm(position)

      sample = wu.Bunch(position=position, lattice=lattice)
      sample = self.mcsym.to_canonical_asu_position(sample, self.compcoms, forceupdate=True, **kw)
      self.check_sample(sample)
      self._startsample = sample
      return sample

   def adjust_spread(self, spread, itrial):
      for k in spread:
         spread[k] *= 0.998
      # spread.cartsd = 1.0
      # spread.rotsd = 0.03
      # spread.latticesd = 2.0

   def label(self):
      return self.kw.architecture + '_' + str.join('_', [c.label.replace('_', '') for c in self.components])

   def new_sample(self, sample, spread, separate_component_sampling=False, **kw):
      kw = self.kw.sub(kw)
      position, lattice = sample.values()
      assert wu.hvalid_norm(position)
      assert position.ndim == 3
      assert lattice.shape == (3, 3)
      if separate_component_sampling:
         icomp = self.component_sample_counter % (self.ncomp + 1)
         if self.ncomp == icomp:
            newpos, newlat = self.mcsym.perturb_lattice(position, lattice, spread, **kw)
            assert wu.hvalid_norm(newpos)
         else:
            newpos, newlat = position, lattice
            perturb = self.components[icomp].perturbation(spread, newpos[icomp], newlat, **kw)
            newpos[icomp] = newpos[icomp] @ perturb
         self.component_sample_counter += 1
      else:
         newpos, newlat = self.mcsym.perturb_lattice(position, lattice, spread, **kw)
         pospert = np.stack([c.perturbation(spread, p, newlat, **kw) for c, p in zip(self.components, newpos)])
         newpos = newpos @ pospert
      assert wu.hvalid_norm(newpos)
      newsample = wu.Bunch(position=newpos, lattice=newlat)
      newsample = self.mcsym.to_canonical_asu_position(newsample, self.compcoms, **kw)
      assert wu.hvalid_norm(newpos)
      self.check_sample(newsample, **kw)
      return newsample

   def check_sample(self, sample, debug=True, **kw):
      assert wu.hvalid_norm(sample.position)
      if not debug: return
      positions, lat = sample.values()
      for icomp, (pos, se) in enumerate(zip(positions, self.mcsym.symelems)):
         secen = wu.hscaled(lat[0, 0], se.cen)
         # ic(icomp, pos[:3, 3], se)
         # axis, ang, cen, hel = wu.haxis_angle_cen_hel_of(pos)
         assert np.allclose(0, wu.hpointlinedis(pos[:, 3], secen, se.axis))
      # assert 0

   def objectivefunc(self, sample, mc_wt_solvfrac=0, **kw):
      kw = self.kw.sub(kw)
      position, lattice = sample.values()
      assert position.shape == (self.ncomp, 4, 4)
      # metainfo = self.metainfo(sample, **kw)
      score = 0
      score += lattice.diagonal().sum() * 0.3
      # score += mc_wt_solvfrac * max(0, metainfo.solvfrac - kw.mc_max_solvfrac)
      # maxdelta = np.max(wu.hnorm(metainfo.deltapos))
      # score += 100 * max(0, maxdelta - kw.mc_tether_components)
      # ic(maxdelta, score)

      framescores, comdist = self.score(sample, **kw)
      compscores = self.mcsym.component_score_summary(framescores)
      # ic(compscores)

      sc = list(sorted(compscores.values()))
      scinter = list(sorted([v for (i, j), v in compscores.items() if i[0] != j[0]]))
      scintra = list(sorted([v for (i, j), v in compscores.items() if i[0] == j[0]]))
      # ic(sc, scinter, scintra)
      # score = np.sum(sc)

      # not enough contact
      if len(sc) < kw.mc_min_contacts:
         comdistsort = np.sort(comdist.reshape(-1))
         return 1_000_000 + comdistsort[kw.mc_min_contacts - 1]

      if len(self.components) > 1 and not kw.mc_disconnected_ok:
         if len(scinter) == 0:
            good = []
            bad = sc
         else:
            rest = list(sorted(scinter[1:] + scintra))
            good = scinter[:1] + rest[:kw.mc_max_contacts - 1]
            bad = rest[kw.mc_max_contacts:]
            # ic(good, bad)
            # assert 0
      else:
         good = sc[:kw.mc_max_contacts]
         bad = sc[kw.mc_max_contacts:]

      # ic(compscores)
      # ic(good, bad)
      # score = score + np.max(good) + 10 * np.sum(np.abs(bad))
      score = score + np.sum(good) + 10 * np.sum(np.abs(bad))
      # ic(good, bad, score)

      return score

   def closest_to_origin(self, position, lattice, output_above_0=False, mc_output_contact_dist=10, **kw):
      frames = self.mcsym.frames(lattice, cells=4)
      position = position.copy()
      sympos = einsum('fij,cjk,ck->fci', frames, position, self.compcoms)
      if output_above_0:
         above0 = np.all(sympos[:, 0, :3] > 0, axis=1)
         frames, sympos = frames[above0], sympos[above0]
      center = np.array([3, 2, 1, 0])
      imin0 = np.argmin(wu.hnorm(sympos[:, 0] - center))
      position[0] = frames[imin0] @ position[0]
      for i in range(1, len(position)):
         body, body2 = self.components[0].body, self.components[i].body
         sympos2 = frames @ position[i]
         ncontact = body.contact_count_ca(body2, position[0], sympos2, mc_output_contact_dist)
         if np.max(ncontact) > 0:
            position[i] = sympos2[np.argmax(ncontact)]
         else:
            position[i] = frames[np.argmin(wu.hnorm(sympos[:, i] - sympos[imin0, 0]))] @ position[i]
      return position

   def contacting_frames(self, position, lattice, mc_output_contact_dist=10, **kw):
      allframes = self.mcsym.frames(lattice, cells='all')
      close = np.zeros(len(allframes), dtype=bool)
      for icomp1, (pos1, comp1) in enumerate(zip(position, self.components)):
         for icomp2, (pos2, comp2) in enumerate(zip(position, self.components)):
            sympos2 = allframes @ pos2
            # ic(pos1.shape, sympos2.shape)
            contact = comp1.body.intersect(comp2.body, pos1, sympos2, mc_output_contact_dist)
            close |= contact
      return allframes[close]

   def dumppdbs(
      self,
      prefix,
      samples,
      dumpasym=True,
      dumpsym=True,
      whichcomp=None,
      rawposition=False,
      cells=None,
      **kw,
   ):
      position, lattice = samples.values()
      assert wu.homog.hvalid_norm(position)
      symtag = self.mcsym.sym.lower().replace(' ', '')
      output_files = list()
      cryst1 = wu.sym.cryst1_line(self.mcsym.sym, lattice)
      if not rawposition:
         position = self.closest_to_origin(position, lattice, **kw)
         assert wu.homog.hvalid_norm(position)
         # frames = self.mcsym.frames(lattice, cells=(-1, 0))
         frames = self.contacting_frames(position, lattice, **kw)
      else:
         if cells is not None:
            frames = self.mcsym.frames(lattice, cells=cells)
         else:
            frames = wu.sym.applylattice(lattice, self.mcsym.closeframes)

      if dumpasym:
         jointcoords = list()
         for icomp, (pos, comp) in enumerate(zip(position, self.components)):
            jointcoords.append(wu.hxformpts(pos, comp.body.coord))
         fname = f'{prefix}_asym.pdb'
         wu.dumppdb(fname, jointcoords, header=cryst1)
         output_files.append(fname)

      if not dumpsym: return
      for icomp, (pos, comp) in enumerate(zip(position, self.components)):
         # ic(pos)
         # ic(np.linalg.det(pos))
         # ic(np.linalg.det(pos[:3, :3]))
         # ic(np.linalg.norm(pos[:3, :3], axis=1))
         assert wu.hvalid(pos)
         assert wu.homog.hvalid_norm(pos)
         # wu.showme(comp.body.coord)
         coords = wu.hxformpts(pos, comp.body.coord)
         # wu.showme(coords)

         fname = f'{prefix}_comp{icomp}_sym.pdb'
         if whichcomp is not None:
            matchcomponents = [c for c in whichcomp if c[0] == icomp]
            frames = self.mcsym.frames(lattice, whichcomp=matchcomponents)
            if frames is None: continue
            ic(frames)
            axis, ang, cen, hel = wu.haxis_angle_cen_hel_of(pos)
            ic(axis, ang, cen, hel)
            ic(pos[:3, 3])
            ic(lattice[0, 0])
            ic(matchcomponents)
            assert 0
            newcen = wu.hscaled(lattice[0][0], self.mcsym.symelems[icomp].cen)

            ic(wu.hpointlinedis(cen, newcen, axis))

            ic(self.mcsym.symelems[icomp].axis)
            # ic(frames.round(2))
            fname += '_whichcomps.pdb'

         # wu.showme(frames)
         # wu.showme(coords, kind='point')
         # ic(position[:, :3, 3])
         # ic(coords[:, :3])
         assert wu.hvalid(frames)
         # ic(frames[0])
         coords = wu.hxformpts(frames, coords)
         # wu.showme(coords, kind='point')

         # assert 0

         wu.dumppdb(fname, coords, header=cryst1)
         output_files.append(fname)
         # wu.showme(coords[:100, :3], is_points=True)
      return output_files

   def sympositions(self, position, lattice, icomp1, icomp2):
      frames0, fmask = self.mcsym.scoreframes_for_component(icomp1, icomp2)
      nframes = len(frames0)
      frames = wu.sym.applylattice(lattice, frames0)
      pos1 = position[icomp1]
      pos2 = wu.hxformx(frames, position[icomp2])
      # pos1 = np.repeat(positions[:, icomp1], nframes, axis=0).reshape(npos, nframes, 4, 4)
      # pos2 = np.repeat(positions[:, icomp2], nframes, axis=0).reshape(npos, nframes, 4, 4)
      # pos2 = wu.hxformx(frames, pos2)
      assert len(pos2)
      return frames, fmask, pos1, pos2

   def metainfo(self, sample, **kw):
      metainfo = wu.Bunch()
      metainfo.solvfrac = self.guess_solvfrac(sample)

      assert self.mcsym.latticetype == 'CUBIC'
      assert 0
      startpos = self._startsample.position[:, :, 3] / self._startsample.lattice[0, 0]
      curpos = sample.position[:, :, 3] / sample.lattice[0, 0]
      metainfo.deltapos = curpos - startpos

      return metainfo

   def score(self, samples, mc_intercomp_only=False, **kw):
      kw = self.kw.sub(kw)
      samples = self._check_samples(samples, **kw)
      positions, lattices = samples.values()
      if len(self.components) == 1: mc_intercomp_only = False

      component_combos = list()
      for icomp1, comp1 in enumerate(self.components):
         for icomp2, comp2 in enumerate(self.components):
            if icomp2 < icomp1: continue  # upper triangle
            component_combos.append(
               (icomp1, comp1, icomp2, comp2, *self.sympositions(positions, lattices, icomp1, icomp2)))

      comdist = 9e6 * np.ones((self.mcsym.nframes, self.ncomp, self.ncomp))
      for icomp1, comp1, icomp2, comp2, frames, fmask, pos1, pos2 in component_combos:
         # don't pull intra contacts together if multicomp
         com = wu.hxform(pos1, comp1.body.com())
         symcom = wu.hxformpts(pos2, comp2.com)
         comdist[fmask, icomp1, icomp2] = wu.hnorm(com - symcom)

      # clash
      scores = np.zeros((self.mcsym.nframes, self.ncomp, self.ncomp))
      for icomp1, comp1, icomp2, comp2, frames, fmask, pos1, pos2 in component_combos:
         compclash = comp1.body.contact_count_ca(comp2.body, pos1, pos2, self.caclashdis)
         scores[fmask, icomp1, icomp2] += 100.0 * compclash
      if scores.sum() > 0: return scores, comdist  # have clashes

      # rpx score
      for icomp1, comp1, icomp2, comp2, frames, fmask, pos1, pos2 in component_combos:
         if icomp1 == icomp2 and mc_intercomp_only: continue
         compscore = self.hscore.scorepos(comp1.body, comp2.body, pos1, pos2, iresl=0, **kw)
         scores[fmask, icomp1, icomp2] -= compscore
      if np.any(scores < 0): return scores, comdist

      return scores, comdist

   def _check_samples(self, sample, debug, **kw):
      if not debug: return sample
      position, lattice = sample.values()
      assert position.ndim == 3 and lattice.shape == (3, 3)
      return sample
      # assert position.shape[-3:] == (self.ncomp, 4, 4)
      # assert lattice.shape[-2:] == (3, 3)
      # position = position.reshape(-1, self.ncomp, 4, 4)
      # lattice = lattice.reshape(-1, 3, 3)
      # assert len(position) == len(lattice)
      # return wu.Bunch(position=position, lattice=lattice)

   def guess_solvfrac(self, sample):
      cellvol = wu.sym.cell_volume(self.mcsym.sym, sample.lattice)
      vol = sum(self.compvols) * wu.sym.copies_per_cell(self.mcsym.sym)
      # ic(sample.lattice.diagonal(), cellvol, self.compvols, vol)
      return 1 - vol / cellvol

class McComponent:
   """Body coupled to sym element"""
   def __init__(
         self,
         coords: np.ndarray,
         symelem: 'SymElem',
         pdbname: str,
         label: str = None,
         index=None,
         bounds=(-1000, 1000),
         mcsym=None,
         **kw,
   ):
      self.kw = wu.Bunch(kw)
      self.symelem = symelem
      if coords.ndim == 3: coords = coords[None, :, :, :]
      self._init_coords = coords.copy()
      if len(coords) > 1:
         assert coords.shape[0] == self.symelem.numops
         ic(self.symelem)
         self._aligned_coords = wu.sym.align(coords, self.symelem)
      else:
         self._aligned_coords = self._init_coords.copy()
      ic(self._aligned_coords.shape)
      self.body = rp.Body(self._aligned_coords[0], **kw)
      # self.body.dump_pdb(f'testbody{index}.pdb')
      self.com = self.body.com()
      self.pdbname = pdbname
      self.label = label or str.join(',', os.path.basename(self.pdbname).split('.')[:-1])
      self.index = index
      self.bounds = bounds
      self.mcsym = mcsym
      assert self.mcsym

   def random_unitcell_position(self, size=1):
      '''random rotation and placement in unit cell'''
      se = self.symelem
      if se.iscyclic:
         lb, ub = self.bounds
         assert lb <= ub
         # ic(self.index, lb, ub)
         shift = np.random.rand(size) * (ub - lb) + lb
         check = np.max(np.abs(self.symelem.axis))
         if np.allclose(check, 1.0): shift = shift % 1.0
         elif np.allclose(check, np.sqrt(2) / 2): shift = shift % np.sqrt(2)
         elif np.allclose(check, np.sqrt(3) / 2): shift[:] = 0.01  # shift % (np.sqrt(3) / 2)
         elif np.allclose(check, np.sqrt(3) / 3): shift = shift % np.sqrt(3)
         else:
            ic(check, self.symelem.axis)
            assert 0

         # way big to avoid boundary effects
         # cell is -0.5 to 0.5
         cen = (se.axis[None] * shift[:, None] + se.cen)
         # if self.index == 1: print(_STUPID, shift[0], cen[:3], flush=True)
         offset = wu.htrans(cen)
         flipangle = np.where(np.random.rand(size) < 0.5, 0, np.pi)

         print('fix flips etc !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!', flush=True)

         # perp = wu.hnormalized(wu.hcross(se.axis, [1, 2, 3]))
         # randrot = wu.hrot(se.axis, np.random.rand(size) * 2 * np.pi, cen)
         # fliprot = wu.hrot(perp, flipangle, cen)
         # pos = wu.hxformx(wu.hxformx(randrot, fliprot), offset)

         pos = offset

         assert np.allclose(0, wu.hpointlinedis(pos[0, :, 3], se.cen, se.axis))
         assert wu.hvalid_norm(pos)

      elif se.isdihedral and se.nfold > 2:
         alignangle = np.where(np.random.rand(size) < 0.5, 0, np.pi / se.nfold)
         mayberot = wu.hrot(se.axis, alignangle, se.cen)
         pos = mayberot @ se.origin

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
         pos = wu.htrans(se.cen) @ oris[np.random.randint(0, 6, size)] @ wu.htrans(-se.cen) @ se.origin
         # pos = wu.htrans(se.cen) @ oris[None, 5] @ wu.htrans(-se.cen) @ se.origin
         # pos = se.origin[None]

      else:
         assert 0

      pos = wu.sym.tounitcell(self.mcsym.lattice_1_1_1_90_90_90, pos)

      assert size == 1
      self._startpos = pos[0]

      # ic(self.mcsym.lattice_1_1_1_90_90_90)
      # ic(pos[:, :3, :3])
      assert wu.hvalid_norm(pos)

      return pos

   def perturbation(self, spread, position, lattice, size=1, **kw):
      '''rotation around origin, should be applied first'''
      kw = self.kw.sub(kw)
      assert size == 1
      if self.symelem.iscompound:
         return np.eye(4)
      rot = np.random.normal(0, spread.rotsd, size)
      x = wu.hrot(self.symelem.axis, rot)

      if kw.mc_tether_components > 10:
         cart = np.random.normal(0, spread.cartsd, size)
         cart = cart * self.symelem.axis
         # # need to move this somewhere that knows abouot the spacegroup
         # assert np.allclose(lattice[0, 1], 0)
         # assert np.allclose(lattice[0, 2], 0)
         # assert np.allclose(lattice[1, 2], 0)
         # assert np.allclose(lattice[1, 0], 0)
         # assert np.allclose(lattice[2, 0], 0)
         # assert np.allclose(lattice[2, 1], 0)
         # assert np.allclose(lattice[0, 0], lattice[1, 1])
         # assert np.allclose(lattice[0, 0], lattice[2, 2])
         # assert position.shape == (4, 4)
         # unitposition = position[:, 3].copy()
         # unitposition2 = unitposition + cart
         # unitposition[:3] /= lattice[0, 0]
         # unitposition2[:3] /= lattice[0, 0]

         # delta = wu.hnorm(self._startpos[:, 3] - unitposition)
         # delta2 = wu.hnorm(self._startpos[:, 3] - unitposition2)
         # # if self.index == 2:
         # ic(delta, delta2)
         # if delta2 > kw.mc_tether_components and delta < delta2:
         #    ic('noshift')
         #    cart = np.array([0, 0, 0, 0])

         x[..., 3] += cart

      return x[0]

   def guess_volume(self):
      return 120 * len(self.body.coord)

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
      self.latticetype = wu.sym.latticetype(sym)
      self.lattice_1_1_1_90_90_90 = wu.sym.lattice_vectors(self.sym, [1, 1, 1, 90, 90, 90], strict=False)
      self.symelems = symelems
      self.allsymelems = wu.sym.symelems(sym)
      # self.unitasucen = wu.hpoint(asucen)
      self.allframes = wu.sym.frames(self.sym, sgonly=True, cells=wu.sym.number_of_canonical_cells(self.sym),
                                     cellgeom='unit')
      self.allopids = np.empty((len(self.allframes), len(symelems)), dtype=np.int32)
      self.allcompids = np.empty((len(self.allframes), len(symelems)), dtype=np.int32)
      self.allopcompids = np.empty((len(self.allframes), len(symelems), len(symelems)), dtype=np.int32)
      for i, ei in enumerate(symelems):
         self.allopids[:, i] = wu.sym.sg_symelem_frame444_opids_dict[self.sym][:, ei.index]
         self.allcompids[:, i] = wu.sym.sg_symelem_frame444_compids_dict[self.sym][:, ei.index]
         for j, ej in enumerate(symelems):
            self.allopcompids[:, i, j] = wu.sym.sg_symelem_frame444_opcompids_dict[self.sym][:, ei.index, ej.index]
      self.reset()

   def reset(self):
      self.celloffset = np.array([0.0, 0.0, 0.0, 0.0])
      self.last_closeframes_cen = np.array([[9e9, 9e9, 9e9, 1]] * len(self.symelems))

   def random_lattice(self, component_volume, mc_cell_bounds, mc_min_solvfrac, mc_max_solvfrac, **kw):
      # assert self.latticetype not in 'MONOCLINIC TRICLINIC'.split()
      # assert self.latticetype == 'CUBIC'
      minsf, maxsf = 9e9, 0
      component_volume *= wu.sym.copies_per_cell(self.sym)
      lb0, ub0 = mc_cell_bounds[0], mc_cell_bounds[1]
      # 1 - sf = (compv / cellv)
      # cellv = compv / (1 - sf)
      mincellside = (component_volume / max(0.0001, (1 - mc_min_solvfrac)))**0.33333
      maxcellside = (component_volume / max(0.0001, (1 - mc_max_solvfrac)))**0.33333
      lb0, ub0 = max(lb0, mincellside), min(ub0, maxcellside)
      # ic(mincellside, maxcellside)
      for i in range(100):  # try many times to meet requirements
         lb, ub = lb0, ub0
         cellgeom = np.random.rand(6) * (ub - lb) + lb
         cellgeom[3:] = np.random.rand(3) * 40 + 80
         cellgeom = wu.sym.full_cellgeom(self.latticetype, cellgeom, strict=False)

         cellvol = wu.sym.cell_volume(self.sym, cellgeom)
         solvfrac = 1 - component_volume / cellvol
         minsf, maxsf = min(minsf, solvfrac), max(maxsf, solvfrac)
         if mc_min_solvfrac <= solvfrac <= mc_max_solvfrac: break
      else:
         raise ValueError(f'Cant make lattice with {mc_min_solvfrac} <= solvfrac <= {mc_max_solvfrac},'
                          f'lowest/highest tries were {minsf}, {maxsf}')
      lattice = wu.sym.lattice_vectors(self.latticetype, cellgeom)

      return lattice

   def perturb_lattice(self, position, lattice, spread, **kw):
      assert position.ndim == 3
      assert lattice.shape == (3, 3)
      cellgeom = np.array(wu.sym.cellgeom_from_lattice(lattice))
      if self.latticetype == 'CUBIC':
         cellgeom[:3] += np.random.normal(0, spread.latticesd)
      elif self.latticetype in ['HEXAGONAL', 'TETRAGONAL']:
         r = np.random.rand()
         if r > 0.3333: cellgeom[:2] += np.random.normal(0, spread.latticesd)
         if r < 0.6666: cellgeom[2] += np.random.normal(0, spread.latticesd)
      elif self.latticetype == 'ORTHORHOMBIC':
         r = np.random.rand()
         if r < 0.25: cellgeom[:3] += np.random.normal(3, spread.latticesd)
         elif r < 0.5: cellgeom[0] += np.random.normal(0, spread.latticesd)
         elif r < 0.75: cellgeom[1] += np.random.normal(0, spread.latticesd)
         else: cellgeom[2] += np.random.normal(0, spread.latticesd)
      elif self.latticetype == 'MONOCLINIC':
         r = np.random.rand()
         if r < 0.2: cellgeom[:3] += np.random.normal(3, spread.latticesd)
         elif r < 0.4: cellgeom[0] += np.random.normal(0, spread.latticesd)
         elif r < 0.6: cellgeom[1] += np.random.normal(0, spread.latticesd)
         elif r < 0.8: cellgeom[2] += np.random.normal(0, spread.latticesd)
         else: cellgeom[4] += np.random.normal(0, spread.latticesd)
      else:
         raise NotImplementedError(f'cant do lattice type {self.latticetype}')

      # ic(lattice)
      # ic(position[:, :3, :3])
      assert wu.hvalid_norm(position)
      position = position.copy()
      position[:, :, 3] = wu.sym.tounitcellpts(lattice, position[:, :, 3])
      # ic(position[:, :3, :3])
      # assert wu.hvalid_norm(position)
      newlattice = wu.sym.lattice_vectors(self.sym, cellgeom)
      position[:, :, 3] = wu.sym.applylatticepts(newlattice, position[:, :, 3])
      assert wu.hvalid_norm(position)

      return position, newlattice

   def setup_component_frames(self, **kw):
      kw = self.kw.sub(kw)
      # self.closeframes = wu.sym.frames(self.sym, sgonly=True, cells=3, asucen=asucen, **kw)
      self.nframes = len(self.closeframes)
      ncomp = len(self.symelems)

      self.scoreframes = np.ones((ncomp, ncomp, self.nframes), dtype=bool)
      self.frames_by_component = list()

      testframes = wu.sym.applylattice(self.lattice_1_1_1_90_90_90, self.closeframes)
      # ic(testframes[:, :3, 3])
      # wu.showme(testframes)
      for icomp, symelem in enumerate(self.symelems):
         self.frames_by_component.append(list())
         assert np.allclose(self.celloffset, 0)
         # ops = wu.htrans(-self.celloffset) @ symelem.operators @ wu.htrans(self.celloffset)
         ops = symelem.operators
         # wu.showme(ops)

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
         # ic()
         # ic(np.sum(~self.scoreframes[icomp, icomp]))
         # ic(icomp, np.sum(~self.scoreframes[icomp, icomp]))
         assert np.sum(~self.scoreframes[icomp, icomp]) == symelem.numops

      norot = np.all(np.isclose(self.closeframes[:, :3, :3], np.eye(3)), axis=(1, 2))
      # self.transframes = self.closeframes[norot]

   def scoreframes_for_component(self, icomp1, icomp2):
      return (
         self.frames_by_component[icomp1][icomp2],
         self.scoreframes[icomp1, icomp2],
      )

   def frames(self, lattice, unitframes=None, whichcomp=None, cells=None):
      if whichcomp is not None:
         assert unitframes is None
         unitframes = list()
         for icomp, jcomp, elemid in whichcomp:
            unitframes.append(self.allframes[self.allopcompids[:, icomp, jcomp] == elemid])
            # ic(icomp, jcomp, elemid, unitframes[-1].shape)
            # ic(unitframes[-1] @ self.symelems[1].cen)
            # assert 0
         if not unitframes: return None
         unitframes = np.concatenate(unitframes)
      if unitframes is None:
         unitframes = self.closeframes
         if cells == 'all': unitframes = self.allframes
         elif cells is not None: unitframes = wu.sym.sgframes(self.sym, cells=cells, cellgeom='unit')
      else:
         assert cells is None
      # nframes = len(unitframes)
      # frames = np.tile(unitframes, (npos, 1, 1)).reshape(npos, nframes, 4, 4)
      frames = wu.sym.applylattice(lattice, unitframes)
      return frames

   def update_closeframes(self, position, lattice, com, mc_framedistcut, forceupdate=False, **kw):
      poscom = einsum('cij,cj->ci', position, com)
      # ic(poscom, com)
      paddingfrac = 3
      dist = [wu.hnorm(poscom[i] - self.last_closeframes_cen[i]) for i in range(len(poscom))]
      close = [d < mc_framedistcut / paddingfrac for d in dist]
      # ic(close)
      if all(close) and not forceupdate:
         return
      # ic('update')
      self.last_closeframes_cen = poscom
      assert len(self.last_closeframes_cen) == len(self.symelems)
      # self.last_closeframes_lattice = np.sum(lattice.diagonal(), axis=0)
      frames = self.frames(lattice, self.allframes)
      close = np.zeros(len(frames), dtype=bool)
      for icomp1, cen in enumerate(poscom):
         for icomp2, cen2 in enumerate(poscom):
            if icomp2 < icomp1: continue
            symcom = einsum('fij,j->fi', frames, cen2)
            dist = wu.hnorm(symcom - cen)
            close |= dist < mc_framedistcut * (1 + 1 / paddingfrac)
      assert np.any(close)
      self.closeframes = self.allframes[close]
      self.closeframes_elemids = self.allopcompids[close]
      # assert 0
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

   def component_score_summary(self, framescores):
      compscores = defaultdict(lambda: 0)
      for (iframe, icomp1, icomp2), score in np.ndenumerate(framescores):
         if score == 0.0: continue
         compid1, compid2 = (icomp1, 0), (icomp2, self.closeframes_elemids[iframe, icomp1, icomp2])
         compscores[compid1, compid2] += score
      return compscores

def component_from_pdb(fname, symelem, index=None, **kw):
   kw = wu.Bunch(kw)
   if isinstance(fname, str):
      pdb = wu.readpdb(fname)
      coords = pdb.ncaco(splitchains=True)
      assert index is not None
      if len(kw.mc_component_bounds) == 0:
         bounds = -9999, 9999
      elif len(kw.mc_component_bounds) == 2:
         bounds = kw.mc_component_bounds
      else:
         bounds = kw.mc_component_bounds[2 * index:2 * index + 2]
      return McComponent(coords, symelem, pdbname=fname, index=index, bounds=bounds, **kw)
      # coords = wu.sym.align(coords, sym='Cx', symelem=symelem)
      # b = rp.Body(coords[0], **kw)
      # return McComponent(b, symelem)
   assert index is None
   return [component_from_pdb(f, s, i, **kw) for i, (f, s) in enumerate(zip(fname, symelem))]

   # sample positions

def _output_line(result):
   compsc = list(sorted((x[1], x[0]) for x in result.compscores.items()))
   compsc += [(0, ((0, 0), (0, 0)))] * 10  # padding
   line = f'{result.score:9.3f} {result.solvfrac:5.3f}'
   for s in compsc[:4]:
      line += f' {s[1][0][0]} {s[1][1][0]} {s[0]:9.3f}'
   return line

#
# score positions
#
# mc check

# xform, cell

if __name__ == '__main__':
   kw = rp.options.get_cli_args()
   if kw.mc_profile:
      import cProfile, pstats
      cProfile.run('main()', filename='/tmp/mcdock_profile')
      p = pstats.Stats('/tmp/mcdock_profile')
      p.sort_stats('tottime')
      p.print_stats(100)
      p.sort_stats('cumtime')
      p.print_stats(100)
      # p.sort_stats('ncalls')
      # p.print_stats(100)

   else:
      main()
