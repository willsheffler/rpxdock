from cppimport import import_hook

from sicdock.util.dilated_int_test import *
from sicdock.util import sanitize_for_pickle
from sicdock import Bunch

class dummyclass:
   pass

def dummyfunc():
   pass

def test_dilated_int():
   assert TEST_dilated_int_64bit()

def test_sanitize_for_pickle():
   raw = test_dilated_int
   san = sanitize_for_pickle(raw)
   assert san.count('.test_dilated_int')

   raw = dict(good=1, bad=test_dilated_int)
   san = sanitize_for_pickle(raw)
   assert san['good'] == 1
   assert san['bad'].count('.test_dilated_int')

   raw = dict(good=1, l=[0, [10, 11, dict(bar=dummyfunc)]], foo=dummyclass, b=Bunch(
      a=1, b=2))
   san = sanitize_for_pickle(raw)
   # print(san)
   assert san['good'] == 1
   assert san['l'][1][1] == 11
   assert san['l'][1][2]['bar'].count('.dummyfunc')
   assert san['foo'].count('.dummyclass')
   assert san['b'].b == 2

if __name__ == '__main__':
   test_sanitize_for_pickle()
