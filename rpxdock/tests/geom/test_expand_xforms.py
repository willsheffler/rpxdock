import numpy as np, rpxdock as rp, rpxdock.homog as hm, pytest

def get_perm_xform(generators, nstep):
   N = len(generators)
   G = np.concatenate([[np.eye(4)], generators])
   idx = np.random.choice(N, nstep)
   xpicks = G[idx]
   x = np.eye(4)
   for i in range(nstep):
      x = x @ xpicks[i]
   return x, idx

def test_expand_xforms_cyclic(ntrials=10):
   for i in range(ntrials):
      N = np.random.randint(2, 20)

      generators = [
         hm.hrot([0, 0, 1], 360 / N, [0, 0, 0]),
         hm.hrot([0, 0, 1], -360 / N, [0, 0, 0]),
      ]
      ex, *_ = rp.geom.expand_xforms(generators, 10, 9e9)
      assert len(ex) is N
      generators = [
         hm.hrot([0, 0, 1], 180.0, [0, 0, 0]),
         hm.hrot([1, 1, 1], 120.0, [0, 0, 0]),
      ]
      ex, *_ = rp.geom.expand_xforms(generators, 8, 9e9)
      assert len(ex) is 12
      # x =
      generators = [
         hm.hrot([0, 0, 1], 90.0, [0, 0, 0]),
         hm.hrot([1, 1, 1], 120.0, [0, 0, 0]),
      ]
      ex, *_ = rp.geom.expand_xforms(generators, 8, 9e9)
      assert len(ex) is 24
      generators = [
         hm.hrot([1, 0, 0], 180.0, [0, 0, 0]),
         hm.hrot([0, 1, 0], 180.0, [0, 0, 0]),
      ]
      ex, *_ = rp.geom.expand_xforms(generators, 8, 9e9)
      assert len(ex) is 4

def do_test_expand_xforms(
   generators,
   nstep,
   radius,
   nexpected,
   radius_intermediate=9e9,
   ntrials=1,
   showme=False,
):
   ex, nseen, ntot = rp.geom.expand_xforms(generators, nstep, radius, radius_intermediate)
   if showme: rp.viz.showme(ex, randpos=2, xyzlen=[1.4, 1.2, 1], scale=1.3)
   if len(ex) != nexpected:
      print('expected', nexpected, 'got', len(ex))
      assert len(ex) is nexpected

   nmissing = 0
   for _ in range(ntrials):
      x, idx = get_perm_xform(generators, nstep)
      if np.linalg.norm(x[:3, 3]) > radius: continue
      c, o = rp.geom.xform_dist2_split(x, ex, 1.0)
      if np.all((c > 0.001) + (o > 0.001)):
         if nmissing < 10:
            print('missing', idx)
            nmissing += 1
   if nmissing > 0:
      print('missing num', nmissing, 'frac', nmissing / 1000_000)
      assert nmissing is 0

def test_expand_xforms_p213(ntrials=10):
   generators = np.array([
      hm.hrot([+0, -1, +1], 180.0, [-1, -1, -1]),
      hm.hrot([-1, -1, +1], 120.0, [-4, +0, -4]),
   ])
   do_test_expand_xforms(
      generators,
      nstep=13,
      radius=5.0,
      nexpected=14,
      radius_intermediate=5.0,
      ntrials=ntrials,
      showme=False,
   )

@pytest.mark.skip
def test_expand_xforms_p4132_2_3(ntrials=10):

   radius = 15.0
   N = 14
   nstep = N
   generators = np.array([
      hm.hrot([-1, 1, 1], 120.0, [-5, -5, 0]),
      hm.hrot([1, -1, 1], 120.0, [0, -5, -5]),
   ])
   do_test_expand_xforms(
      generators,
      nstep=13,
      radius=5.0,
      nexpected=14,
      radius_intermediate=5.0,
      ntrials=ntrials,
      showme=False,
   )

if __name__ == '__main__':
   test_expand_xforms_cyclic(ntrials=10)
   test_expand_xforms_p213(ntrials=10)
   # test_expand_xforms_p4132_2_3()
   print('DONE')

# (232, 4, 4)
# (232,)
# 112

# CPP N=50 seenit.size 805 uniq 805 (matches N=20)
# count r1.1 30
# count r1.5 84
# count r1.9 214
# count r2.1 232
