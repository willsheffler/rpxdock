import threading, numpy as np
from os.path import join
from cppimport import import_hook
from rpxdock.util.dilated_int_test import *
from rpxdock.util import sanitize_for_pickle, load_threads, dump, can_pickle, num_digits
from rpxdock import Bunch

class dummyclass:
   pass

def dummyfunc():
   pass

def test_dilated_int():
   assert TEST_dilated_int_64bit()

def test_load_threads(tmpdir):
   dump('a', join(tmpdir, 'a'))
   dump('b', join(tmpdir, 'b'))
   dump('c', join(tmpdir, 'c'))
   dump('d', join(tmpdir, 'd'))
   fnames = [join(tmpdir, _) for _ in 'abcd']
   x = load_threads(fnames, nthread=4)
   assert x == list('abcd')

def test_sanitize_for_pickle():
   raw = test_dilated_int
   san = sanitize_for_pickle(raw)
   assert san.count('.test_dilated_int')

   raw = dict(good=1, bad=test_dilated_int)
   san = sanitize_for_pickle(raw)
   assert san['good'] == 1
   assert san['bad'].count('.test_dilated_int')

   raw = dict(good=1, l=[0, [10, 11, dict(bar=dummyfunc)]], foo=dummyclass, b=Bunch(a=1, b=2))
   san = sanitize_for_pickle(raw)
   # print(san)
   assert san['good'] == 1
   assert san['l'][1][1] == 11
   assert san['l'][1][2]['bar'].count('.dummyfunc')
   assert san['foo'].count('.dummyclass')
   assert san['b'].b == 2

def test_can_pickle():
   assert can_pickle(7)
   assert not can_pickle([threading.Lock()])

def test_num_digits():
   assert num_digits(0) == 1
   assert num_digits(999) == 3
   assert num_digits(1000) == 4
   assert num_digits(-1000) == 5
   N = 100
   rand = np.exp(np.random.rand(N) * 30).astype('i8')
   rand[np.random.rand(N) < 0.5] *= -1
   print(rand)
   ndig = num_digits(rand)
   for r, d in zip(rand, ndig):
      assert d == len(str(r))

if __name__ == '__main__':
   import tempfile
   # test_load_threads(tempfile.mkdtemp())
   # test_sanitize_for_pickle()
   # test_can_pickle()
   test_num_digits()
