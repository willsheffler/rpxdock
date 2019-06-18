import time
from rpxdock.util import Cache, GLOBALCACHE, Timer

_ncalc = 0

def sum_(*args):
   global _ncalc
   _ncalc += 1
   return sum(args)

def test_cache():
   cache = Cache()
   # with Timer() as t:
   #    for i in range(1000):
   #       cache(time.sleep, 0.001)
   #       t.checkpoint('timesleep')
   # assert t.timesleep[0] >= 0.001
   # assert t.mean.timesleep < 0.0001

   global _ncalc
   assert cache(sum_, 1, 2) == 3
   assert _ncalc == 1
   assert cache(sum_, 1, 2) == 3
   assert _ncalc == 1
   assert cache(sum_, 2, 1) == 3
   assert _ncalc == 2
   assert cache(sum_, 2, 1) == 3
   assert _ncalc == 2
   assert cache(sum_, 2, 1, 2.5) == 5.5
   assert _ncalc == 3
   assert cache(sum_, 2, 1, 2.5) == 5.5
   assert _ncalc == 3

if __name__ == '__main__':
   test_cache()
