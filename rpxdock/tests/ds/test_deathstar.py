import os, pytest
import rpxdock as rp, willutil as wu, numpy as np

def main():
   # test_deathstar_onecomp()
   # helper_test_deathstar('icos', 'c3', showme=True)
   _test_deathstar_mc('icos', 'c3')
   print('DONE')

def _test_deathstar_onecomp():
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

   # fcap = '/home/sheffler/debug/deathstar/cage_examples/I3_AK/I3ak_plusone_center_fix.pdb'
   # fcap = 'test.pdb'
   # capbody = rp.get_body_cached(fcap)
   # capbody.coord, capbody.stub = sort_ncac_coords(capbody.coord, capbody.stub)
   # capbody.coord = sort_ncac_coords(capbody.coord)
   # capbody.dump_pdb('test.pdb')
   # assert 0

   rad = np.linalg.norm(body.bvh_bb.com())
   xaln = wu.align_vector([1, 1, 1], [0, 0, 1])
   xaln[2, 3] = -rad

   # fixes neighbors
   origin = wu.hrot([0, 0, 1], np.pi * 2 / 3 + wu.hangle([1, 1, 1], [1, 1, 0]))
   # origin = wu.hrot([0, 0, 1], wu.hangle([1, 1, 1], [1, 1, 0]))
   origin[2, 3] = rad

   body = rp.get_body_cached(forig, csym, xaln)

   # flaser = '/home/sheffler/debug/deathstar/I3ak_orig_expanded.pdb'
   # flaser = forig
   # laser = rp.rp.get_body_cached(flaser)
   # laser = laser.copy_xformed(xaln)
   # laser = laser.copy_with_sym(csym)

   # cap_xform = wu.htrans([0, 9, 0]) @ wu.hrot([1, 0, 0], -3, degrees=True)
   # caporigin = wu.hconstruct(
   # np.array([(0.998264, -0.055683, 0.019177), (0.052239, 0.987573, 0.148226),
   # (-0.027193, -0.146967, 0.988768)]), np.array((8.401961, -17.734983, 5.701402)))

   # caporigin = wu.hconstruct(
   # np.array([(0.998999, 0.039269, -0.021442), (-0.042080, 0.987471, -0.152085),
   # (0.015201, 0.152835, 0.988135)]), np.array([-7.082527, 19.054423, -2.096715]))
   # fcap = 'test_cen.pdb'
   # alnframe = wu.hconstruct(
   #    np.array([(0.310026, 0.868998, 0.385651), (-0.677772, -0.082444, 0.730636),
   #              (-0.666716, 0.487899, -0.563422)]), np.array((85.476997, 25.174000, 36.908001)))
   # cenframe = wu.hconstruct(
   #    np.array([(0.334448, 0.881423, 0.333524), (-0.751964, 0.036265, 0.658206),
   #              (-0.568063, 0.470933, -0.674927)]), np.array((93.037003, 17.063000, 36.169998)))
   # relframe = wu.hinv(cenframe) @ alnframe

   # laser = laser.copy_xformed(caporigin)

   # wu.showme(laser, pos=np.eye(4))
   # wu.showme(laser, pos=relframe)
   # assert 0
   # fcap = '/home/sheffler/debug/deathstar/cage_examples/I3_AK/I3ak_plusone_center_fix.pdb'
   # laser = rp.get_body_cached(fcap, csym, xaln)
   laser = body
   # ds = rp.DeathStar(body, laser, sym, csym, origin=origin, capcen=cenframe, capaln=alnframe)
   # ds = rp.DeathStar(body, laser, sym, csym, origin=origin)

   #

   # origin = wu.hrot([0, 0, 1], -12, degrees=True) @ origin
   # origin = wu.htrans([0, 0, 30.5]) @ origin

   #

   print('------ origin ------')
   print(origin)
   print('--------------------')

   # assert 0

   # origin[:, 3] += [0, 0, 100]
   # print(origin)
   # assert 0

   ds = rp.DeathStar(body, laser, sym, csym, origin=origin, capxform=wu.hrot([0, 0, 1], 240))

   # if not os.path.exists(
   #       '/home/sheffler/debug/deathstar/cage_examples/I3_AK/I3ak_orig.dstariface.npy'):
   #    np.save('/home/sheffler/debug/deathstar/cage_examples/I3_AK/I3ak_orig.dstariface.npy',
   #            ds.frames)
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

def _test_deathstar_mc(
   sym,
   csym,
   showme=True,
   startover=True,
):

   showopts = wu.Bunch(
      saveview=False,
      connsphere=8.0,
      conncyl=4.5,
      show_aligned_ifaces=False,
      linewidth=5,
      showaxis=True,
      whole=True,
      showcaporigin=False,
      showcap=False,
      showopening=True,
   )

   ds = helper_get_deathstar(sym, csym)
   if not startover:
      import glob
      g = sorted(glob.glob('dstar_mc/dstar_mc_best_*.pickle'))
      print('-' * 80)
      if g:
         print('loading', g[0])
         ds = rp.load(g[0])
         # ds = rp.load('dstar_mc/dstar_mc_best_088.247.pickle')
         ds.set_dofs(ds.dofs())
         # ds.set_dofs(wu.hrand(len(ds.dofs()), cart_sd=0.5, rot_sd=0.01) @ ds.dofs())
      else:
         print('starting over')
         # assert 0
      print('-' * 80)
      if 1:
         # if not hasattr(ds, 'begnbr'):
         # ds.begnbr = 1
         # ds.ref_iface_idx = -1
         sc = ds.scoredofs(ds.dofs(), tether=True)
         tmp = ds.scoredofs(ds.dofs(), tether=False)
         xy = ds.getspread()
         print(f'START  {sc:6.3f} {xy/tmp:6.3f} {tmp:6.3f} {xy:6.3f} {ds.ncontacts():3}',
               flush=True)
   ds.origframes = np.load(
      '/home/sheffler/debug/deathstar/cage_examples/I3_AK/I3ak_orig.dstariface.npy')

   if showme:
      wu.showme(ds, **showopts)
      assert 0

   # T = wu.Timer()
   T = ds.timer
   # print(ds.scoredofs(ds.dofs()))
   Ninner = 1_000
   Nouter = 1
   # Nouter = 1
   # temp = 0.03
   temp = 0.3
   cart_sd = 0.02
   rot_sd = cart_sd / 30
   symptrbfrac = 0.0
   # temp = 0.002
   # cart_sd = 0.03
   # rot_sd = 0.0009
   # cart_sd, rot_sd = 0.0, 0.0

   from time import perf_counter

   mc = wu.MonteCarlo(ds.scoredofs, temperature=temp, debug=False, timer=T)
   mc.try_this(ds.dofs())
   ds.timer = wu.Timer()
   nframes = 0

   if showme:
      tag = 'start'
      wu.showme(ds, png=f'dstarmc_{rp.ds.spread_weight}_{tag}.png', ray=True, **showopts)
      # wu.showme(
      # ds, png=f'dstarmc_{rp.ds.spread_weight}_{tag}_iface.png', ray=True,
      # **showopts.sub(show_aligned_ifaces=True, showcap=False, showopening=False,
      # showbody=False, showaxis=0, connsphere=0, conncyl=0))

   rp.dump(ds, f'dstarmc_{rp.ds.spread_weight}_{tag}.pickle')

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
         xrand[:, i] = wu.hrot([0, 0, 1], np.random.normal(size=Ninner) * rot_sd * 2)
         xrand[:, i, 2, 3] = np.random.normal(size=Ninner) * cart_sd * 2
      tstart = perf_counter()
      dofs = ds.dofs()

      for inner in range(1, Ninner):
         # wu.PING()
         i = Ninner * outer + inner

         if i % 100 == 0 and mc.bestconfig is not None:
            dofs = mc.bestconfig
         # wu.PING()
         dofs0 = dofs.copy()
         dofs = dofs @ xrand[inner]
         # also move symmetrically
         dofs = dofs @ wu.hrand(len(dofs), cart_sd=cart_sd * symptrbfrac,
                                rot_sd=rot_sd * symptrbfrac)

         # print(origdiff, flush=True)
         # assert 0

         # assert 0
         accepted = mc.try_this(dofs)
         if i % 1000 == 0:
            tmp = ds.scoredofs(dofs, tether=False)

            # rimcom = ds.laser.symcom(ds.frames[1])[0, 1, :2, 3]
            spread = ds.getspread()
            # if showme: wu.showme(ds, delprev=True)

         if not accepted:
            dofs = dofs0

         if mc.new_best_last:
            dofdiff = ds.scoredofs(dofs, tether=False)
            # rimcom = ds.laser.symcom(ds.frames[1])[0, 1, :2, 3]
            spread = ds.getspread()
            c2diff = ds.symdiff_c2()
            symdiff = ds.symdiff()
            # origdiff = wu.hdiff(ds.origframes[[4]], ds.frames[[4]], ds.lever).squeeze()
            print(
               f'B{i:9,} {mc.naccept/(i+1):6.3f} {mc.best:9.3f} | ',
               f'{ds.ncontacts():3}{int(inner/(perf_counter()-tstart)):5} |',
               f'{dofdiff:6.3f} {spread:6.3f} |',
               f'{symdiff:6.3f} {c2diff:6.3f} |',
               f'{spread/dofdiff:6.3f} {c2diff/dofdiff:7.3f}',
               flush=True,
            )
            if showme and i % 1 == 0 and i > 5000:
               wu.showme(ds, delprev=True, png=f'dstarmov/frame_{nframes:04}.png', **showopts)
               # wu.showme(
               # ds, delprev=True, png=f'dstarmov/iface_{nframes:04}.png',
               # **showopts.sub(show_aligned_ifaces=True, showcap=False, showopening=False,
               # showbody=False, showaxis=0, connsphere=0, conncyl=0))

               nframes += 1
               rp.dump(ds, f'dstar_mc/dstar_mc_best_{mc.best+1000000:06.3f}.pickle')

            # ds.dump_pdb('dstar_best_hull')

      T.report(file='./dsprof.txt')

   if showme:
      tag = 'end'
      wu.showme(ds, png=f'dstarmc_{rp.ds.spread_weight}_{tag}.png', ray=True, **showopts)
      # wu.showme(
      # ds, png=f'dstarmc_{rp.ds.spread_weight}_{tag}_iface.png', ray=True,
      # **showopts.sub(show_aligned_ifaces=True, showcap=False, showopening=False,
      # showbody=False, showaxis=0, connsphere=0, conncyl=0))
   rp.dump(ds, f'dstarmc_{rp.ds.spread_weight}_{tag}.pickle')

   print(T)
   # print(repr(ds.dofs()))
   # ds.set_dofs(mc.startconfig)
   # wu.showme(ds, saveview=False)

   # if i % 10 == 0:

if __name__ == '__main__':
   main()
