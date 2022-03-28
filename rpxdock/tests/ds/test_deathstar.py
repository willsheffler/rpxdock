import rpxdock as rp, willutil as wu, numpy as np
import pytest

def main():
   # test_deathstar_onecomp()
   # helper_test_deathstar('icos', 'c3', showme=True)
   test_deathstar_mc('icos', 'c3')
   print('DONE')

def test_deathstar_onecomp():
   helper_test_deathstar('icos', 'c3')
   helper_test_deathstar('icos', 'c5')
   helper_test_deathstar('icos', 'c2')
   helper_test_deathstar('oct', 'c4')
   helper_test_deathstar('oct', 'c3')
   helper_test_deathstar('oct', 'c2')
   helper_test_deathstar('tet', 'c3')
   helper_test_deathstar('tet', 'c2')

def helper_get_deathstar(sym, csym):
   forig = '/home/sheffler/debug/deathstar/cage_examples/I3_AK/I3ak_orig.pdb'
   body = rp.get_body_cached(forig)

   rad = np.linalg.norm(body.bvh_bb.com())
   xaln = wu.align_vector([1, 1, 1], [0, 0, 1])
   xaln[2, 3] = -rad
   origin = wu.hrot([0, 0, 1], wu.angle([1, 1, 1], [1, 1, 0]))
   origin[2, 3] = rad

   # body = body.copy_xformed(xaln)
   # body = body.copy_with_sym(csym)
   body = rp.get_body_cached(forig, csym, xaln)

   # flaser = '/home/sheffler/debug/deathstar/I3ak_orig_expanded.pdb'
   flaser = forig
   print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
   print('!!!!!!!!!!!!!! put test pdbs into repo      !!!!!!!!!!!!!!!!!!!!!!!')
   print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
   # laser = rp.rp.get_body_cached(flaser)
   # laser = laser.copy_xformed(xaln)
   # laser = laser.copy_with_sym(csym)
   laser = rp.get_body_cached(flaser, csym, xaln)
   ds = rp.DeathStar(body, laser, sym, csym, origin=origin)
   return ds

def helper_test_deathstar(sym, csym, showme=False):
   ds = helper_get_deathstar(sym, csym)

   # check asym unit is complete
   flb, fub, fnum = wu.sym.symunit_bounds(sym, csym)
   nfold = int(csym[1:])
   assert len(ds.asymunit) == (len(wu.sym.frames(sym)) / nfold - fnum) / nfold
   for x in wu.hrot([0, 0, 1], np.arange(1, nfold) / nfold * 2 * np.pi):
      for i in range(len(ds.asymunit)):
         for j in range(i):
            assert not np.allclose(ds.asymframes[i], ds.asymframes[j], atol=1e-6)

   # check symmetric dependencies
   for a, b in ds.follows.items():
      assert a not in ds.asymunit
      assert b in ds.asymunit
      xsym = wu.hrot([0, 0, 1], 1 / nfold * 2 * np.pi)
      assert np.allclose(xsym @ ds.frames[b], ds.frames[a], atol=1e-6)

   # check all iface distances same(ish) for symmetrycal system
   d = list()
   for (icage, jcage), (icyc, jcyc) in zip(ds.neighbors, ds.nbrs_internal):
      d.append(ds.hull.symcomdist(ds.frames[icage], ds.frames[jcage])[0, icyc, 0, jcyc])
   assert np.max(d) < np.min(d) + 0.1

   if showme:
      wu.showme(ds, whole=False, asymframes=False, delprev=False)

def test_deathstar_mc(sym, csym, showme=True):

   ds = helper_get_deathstar(sym, csym)
   import glob
   g = sorted(glob.glob('dstar_mc/dstar_mc_best_*.pickle'))
   print('-' * 80)
   if g:
      print('loading', g[0])
      ds = rp.load(g[0])
      # ds.set_dofs(wu.hrand(len(ds.dofs()), cart_sd=0.5, rot_sd=0.01) @ ds.dofs())
   else:
      print('starting over')
   print('-' * 80)
   if 1:
      sc = ds.scoredofs(ds.dofs(), tether=True)
      tmp = ds.scoredofs(ds.dofs(), tether=False)
      xy = ds.getspread()
      print(f'START  {sc:9.5f} {xy/tmp:9.5f} {tmp:9.5f} {xy:7.3f} {ds.ncontacts():3}', flush=True)

   if showme:
      wu.showme(ds, saveview=False, whole=True, linewidth=1)
   # assert 0

   T = wu.Timer()
   # print(ds.scoredofs(ds.dofs()))
   Ninner = 100_000
   Nouter = 100_000
   # temp = 0.03
   temp = 0.2
   cart_sd = 0.005
   rot_sd = cart_sd / 30
   # temp = 0.002
   # cart_sd = 0.03
   # rot_sd = 0.0009
   # cart_sd, rot_sd = 0.0, 0.0

   T.checkpoint('genrand')

   from time import perf_counter

   mc = wu.MonteCarlo(ds.scoredofs, temperature=temp)
   mc.try_this(ds.dofs(), timer=T)
   for outer in range(Nouter):
      # mc.temperature = temp0 * mc.best**2 / 17
      # cart_sd = cart_sd0 * mc.best**2 / 17
      # rot_sd = rot_sd0 * mc.best**2 / 17
      # xrandbig = wu.hrand(len(ds.dofs()), cart_sd=10 * cart_sd, rot_sd=10 * rot_sd)
      # xrandbig[-1] = wu.hrot([0, 0, 1], np.random.normal() * rot_sd * 30)
      # xrandbig[-1, 2, 3] = np.random.normal() * cart_sd * 30
      # ds.set_dofs(xrandbig @ ds.dofs())

      xrand = wu.hrand((Ninner, len(ds.dofs())), cart_sd=cart_sd, rot_sd=rot_sd)
      for i in [-1]:
         xrand[:, i] = wu.hrot([0, 0, 1], np.random.normal(size=Ninner) * rot_sd * 3)
         xrand[:, i, 2, 3] = np.random.normal(size=Ninner) * cart_sd * 3
      tstart = perf_counter()
      for inner in range(1, Ninner):

         i = Ninner * outer + inner
         dofs = ds.dofs()
         if i % 1000 == 0: dofs = mc.bestconfig
         dofs = dofs @ xrand[inner]
         T.checkpoint('purturb')
         accepted = mc.try_this(dofs, timer=T)

         if i % 1000 == 0:
            tmp = ds.scoredofs(dofs, tether=False)
            # rimcom = ds.laser.symcom(ds.frames[1])[0, 1, :2, 3]
            xy = ds.getspread()
            print(
               f'ACCEPT {i:10,} {mc.naccept/i:9.5f} {mc.low:9.5f} {mc.best:9.5f} | {xy/tmp:9.5f} {tmp:9.5f} {xy:7.3f} {ds.ncontacts():3} {int(inner/(perf_counter()-tstart)):5}',
               flush=True)
            # if showme: wu.showme(ds, whole=True, delprev=True)
         T.checkpoint('trythis')
         if accepted:
            ds.set_dofs(dofs)
         if mc.new_best_last:
            tmp = ds.scoredofs(dofs, tether=False)
            # rimcom = ds.laser.symcom(ds.frames[1])[0, 1, :2, 3]
            xy = ds.getspread()
            print(
               f'BEST   {i:10,} {mc.naccept/i:9.5f} {mc.low:9.5f} {mc.best:9.5f} | {xy/tmp:9.5f} {tmp:9.5f} {xy:7.3f} {ds.ncontacts():3} {int(inner/(perf_counter()-tstart)):5}',
               flush=True)
            if showme: wu.showme(ds, whole=True, delprev=True)
            rp.dump(ds, f'dstar_mc/dstar_mc_best_{mc.best+100:07.3f}.pickle')

         T.checkpoint('iter')  # T.checkpoint('accept')

   T.checkpoint('done')
   # print(repr(ds.dofs()))
   # ds.set_dofs(mc.startconfig)
   # wu.showme(ds, saveview=False)
   print(T)
   # if i % 10 == 0:

if __name__ == '__main__':
   main()
