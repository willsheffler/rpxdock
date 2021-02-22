import numpy as np, rpxdock as rp, rpxdock.homog as hm, pytest

def _ex_process_generators(generators):
   generators = np.asarray(generators)
   # flatten
   # filter dups
   # print('warning generators not processed')
   return generators

def _ex_get_cen_radius(generators):
   origins = generators[:, :3, 3]
   len_2 = np.sum(origins**2, axis=1)
   radius = np.sqrt(np.mean(len_2))
   cen = np.mean(origins, axis=0)
   return radius, cen

def expand_xforms_rand(
      generators,
      trials=100,
      depth=10,
      radius=None,
      radius_mult=8,
      cen=[0, 0, 0],
      reset_radius_ratio=1.5,  # unused atm
):

   generators = _ex_process_generators(generators)

   radius0, cen0 = _ex_get_cen_radius(generators)
   if radius is None: radius = radius0 * radius_mult + 1.0
   if cen is 'auto': cen = cen0

   # multiply out xforms and bin
   binner = rp.xbin.Xbin(0.1654234, 1.74597824, 107)
   phmap = rp.phmap.PHMap_u8u8()

   frames = np.zeros((depth, trials, 4, 4))
   frames[:, :, :, :] = np.eye(4)
   for idepth in range(depth):
      which = np.random.choice(len(generators), trials)
      xdelta = generators[which]
      if idepth is 0:
         frames[idepth] = xdelta
      else:
         frames[idepth] = xdelta @ frames[idepth - 1]
      bins = binner[frames[idepth]]
      phmap[bins] = idepth * trials + np.arange(trials, dtype='u8')

   keys, unique_idx = phmap.items_array()
   idepth = unique_idx // trials
   itrial = unique_idx % trials

   unique_key_frames = frames[idepth, itrial]

   keep = list()
   for i, x in enumerate(unique_key_frames):
      dist, ang = rp.geom.xform_dist2_split(x, unique_key_frames, 1.0)
      # print(i, dist, ang)
      if 1 == np.sum(np.sqrt(dist**2 + ang**2) < 0.0001):
         keep.append(x)
   unique_frames = np.array(keep)

   centers = unique_frames[:, :3, 3]
   dist2cen = np.linalg.norm(centers - cen, axis=1)
   unique_frames = unique_frames[dist2cen <= radius]

   return unique_frames, None

def get_perm_xform(generators, nstep):
   N = len(generators)
   G = np.concatenate([[np.eye(4)], generators])
   idx = np.random.choice(N, nstep)
   xpicks = G[idx]
   x = np.eye(4)
   for i in range(nstep):
      x = x @ xpicks[i]
   return x, idx

def _test_expand_xforms_various_count(expand_xforms_func, trials=3):
   for i in range(trials):

      N = np.random.randint(2, 20)
      generators = [
         hm.hrot([0, 0, 1], 360 / N, [0, 0, 0]),
         hm.hrot([0, 0, 1], -360 / N, [0, 0, 0]),
      ]
      ex, *_ = expand_xforms_func(generators, depth=100, trials=100, radius=0.1)

      generators = [
         hm.hrot([0, 0, 1], 180.0, [0, 0, 0]),
         hm.hrot([1, 1, 1], 120.0, [0, 0, 0]),
      ]
      ex, *_ = expand_xforms_func(generators, depth=100, trials=100, radius=9e9)
      assert len(ex) is 12

      generators = [  # O
         hm.hrot([0, 1, 1], 180.0, [0, 0, 0]),
         hm.hrot([1, 1, 1], 120.0, [0, 0, 0]),
      ]
      ex, *_ = expand_xforms_func(generators, depth=100, trials=100, radius=9e9)
      assert len(ex) is 24

      generators = [  # D2
         hm.hrot([1, 0, 0], 180.0, [0, 0, 0]),
         hm.hrot([0, 1, 0], 180.0, [0, 0, 0]),
      ]
      ex, *_ = expand_xforms_func(generators, depth=50, trials=50, radius=9e9)
      assert len(ex) is 4

      generators = [
         hm.hrot([+0, -1, +1], 180.0, [-1, -1, -1]),
         hm.hrot([-1, -1, +1], 120.0, [-4, +0, -4]),
         hm.hrot([-1, -1, +1], 240.0, [-4, +0, -4]),
      ]
      frames, meta = expand_xforms_func(
         generators,
         depth=500,
         trials=500,
         radius=6.1,
         cen=[0, 0, 0],
         reset_radius_ratio=9e9,
      )
      assert len(frames) == 51, f"got: {len(frames)}"

      frames, meta = expand_xforms_func(
         generators,
         depth=700,
         trials=700,
         radius=8.1,
         cen=[0.001, 0.023, 0.07123],  # tiny perterb
         reset_radius_ratio=9e9,
      )
      assert len(frames) == 105, f"got: {len(frames)}"

def test_expand_xforms_various_count_cpp(trials=3):
   _test_expand_xforms_various_count(rp.geom.expand_xforms_rand, trials)

def _test_expand_xforms_various_count_py(trials=3):
   _test_expand_xforms_various_count(expand_xforms_rand, trials)

def do_test_expand_xforms(
   generators,
   nstep,
   radius,
   nexpected,
   radius_intermediate=9e9,
   trials=1,
   showme=False,
   expand_xforms_func=rp.geom.expand_xforms_rand,
):
   ex, _ = expand_xforms_func(generators, depth=4 * nstep, radius=radius, trials=10000)
   if showme: rp.viz.showme(ex, randpos=2, xyzlen=[1.4, 1.2, 1], scale=1.3)
   if len(ex) != nexpected:
      print('expected', nexpected, 'got', len(ex))
      assert len(ex) is nexpected

   nmissing = 0
   for _ in range(trials):
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

def test_expand_xforms_p213_rand(trials=10):
   generators = np.array([
      hm.hrot([+0, -1, +1], 180.0, [-1, -1, -1]),
      hm.hrot([-1, -1, +1], 120.0, [-4, +0, -4]),
   ])
   do_test_expand_xforms(
      generators,
      nstep=13,
      radius=5.0,
      nexpected=15,
      radius_intermediate=5.0,
      trials=trials,
      showme=False,
   )

@pytest.mark.skip
def test_expand_xforms_p4132_2_3(trials=10):

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
      trials=trials,
      showme=False,
   )

if __name__ == '__main__':
   t = rp.Timer().start()
   # test_expand_xforms_rand()
   t.checkpoint('start')
   test_expand_xforms_various_count_cpp(trials=1)
   t.checkpoint('cpp')
   _test_expand_xforms_various_count_py(trials=1)
   t.checkpoint('py')
   print(t)
   print(t.mean.py / t.mean.cpp)

   # test_expand_xforms_p213_rand(trials=1)
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
