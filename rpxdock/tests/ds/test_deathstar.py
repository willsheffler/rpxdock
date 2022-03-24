import rpxdock as rp, willutil as wu, numpy as np
import pytest

def main():
   test_deathstar_onecomp()
   # helper_test_deathstar('icos', 'c3', showme=True)
   print('DONE')

def test_deathstar_onecomp():

   t = wu.Timer()
   helper_test_deathstar('icos', 'c3')
   t.checkpoint('icos_c3')
   helper_test_deathstar('icos', 'c5')
   t.checkpoint('icos_c5')
   helper_test_deathstar('icos', 'c2')
   t.checkpoint('icos_c2')
   helper_test_deathstar('oct', 'c4')
   t.checkpoint('oct_c4')
   helper_test_deathstar('oct', 'c3')
   t.checkpoint('oct_c3')
   helper_test_deathstar('oct', 'c2')
   t.checkpoint('oct_c2')
   helper_test_deathstar('tet', 'c3')
   t.checkpoint('tet_c3')
   helper_test_deathstar('tet', 'c2')
   t.checkpoint('tet_c2')
   # t.report()

def helper_test_deathstar(sym, csym, showme=False):
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

   fcap = '/home/sheffler/debug/deathstar/I3ak_orig_expanded.pdb'
   # cap = rp.rp.get_body_cached(fcap)
   # cap = cap.copy_xformed(xaln)
   # cap = cap.copy_with_sym(csym)
   cap = rp.get_body_cached(fcap, csym, xaln)

   ds = rp.DeathStar(body, cap, sym, csym, origin=origin)
   # wu.viz.showme(ds)

   flb, fub, fnum = wu.sym.symunit_bounds(sym, csym)
   nfold = int(csym[1:])
   assert len(ds.asymunit) == (len(wu.sym.frames(sym)) / nfold - fnum) / nfold

   for x in wu.hrot([0, 0, 1], np.arange(1, nfold) / nfold * 2 * np.pi):
      for i in range(len(ds.asymunit)):
         for j in range(i):
            assert not np.allclose(ds.asymframes[i], ds.asymframes[j], atol=1e-3)
   for a, b in ds.follows.items():
      assert a not in ds.asymunit
      assert b in ds.asymunit
      xsym = wu.hrot([0, 0, 1], 1 / nfold * 2 * np.pi)
      # wu.viz.showme(np.array([ds.frames[b], ds.frames[a]]), xyzscale=10)
      assert np.allclose(xsym @ ds.frames[b], ds.frames[a], atol=1e-6)

   d = list()
   for (m, n), (a, b) in zip(ds.neighbors, ds.nbrs_internal):
      d.append(body.symcomdist(ds.frames[m], ds.frames[n])[0, a, 0, b])
   assert np.max(d) < np.min(d) + 0.1

   if showme:
      wu.viz.showme(ds, allframes=False, asymframes=False, delprev=False)

if __name__ == '__main__':
   main()
