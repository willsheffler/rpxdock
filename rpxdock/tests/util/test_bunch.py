import _pickle
from argparse import Namespace
from rpxdock.util import Bunch

def test_bunch_pickle(tmpdir):
   x = Bunch(dict(a=2, b="bee"))
   x.c = "see"
   with open(tmpdir + "/foo", "wb") as out:
      _pickle.dump(x, out)

   with open(tmpdir + "/foo", "rb") as inp:
      y = _pickle.load(inp)

   assert x == y
   assert y.a == 2
   assert y.b == "bee"
   assert y.c == "see"

def test_bunch_init():
   b = Bunch(dict(a=2, b="bee"))
   b2 = Bunch(b)
   b3 = Bunch(c=3, d="dee", **b)

   assert b.a == 2
   assert b.b == "bee"
   assert b.missing is None

   assert b.a == 2
   assert b.b == "bee"
   assert b.missing is None

   assert b3.a == 2
   assert b3.b == "bee"
   assert b3.missing is None
   assert b3.c is 3
   assert b3.d is "dee"

   foo = Namespace(a=1, b='c')
   b = Bunch(foo)
   assert b.a == 1
   assert b.b == 'c'
   assert b.missing is None

def test_bunch_sub():
   b = Bunch(dict(a=2, b="bee"))
   assert b.b == "bee"
   b2 = b.sub(b="bar")
   assert b2.b == "bar"
   b3 = b.sub({"a": 4, "d": "dee"})
   assert b3.a == 4
   assert b3.b == "bee"
   assert b3.d == "dee"
   assert b3.foobar is None

   assert 'a' in b
   b4 = b.sub(a=None)
   assert not 'a' in b4
   assert 'b' in b4

def test_bunch_items():
   b = Bunch(dict(item='item'))
   b.attr = 'attr'
   assert len(list(b.items())) == 2
   assert list(b) == ['item', 'attr']
   assert list(b.keys()) == ['item', 'attr']
   assert list(b.values()) == ['item', 'attr']

def test_bunch_add():
   b1 = Bunch(dict(a=2, b="bee", mergedint=4, mergedstr='b1'))
   b2 = Bunch(dict(a=2, c="see", mergedint=3, mergedstr='b2'))
   b1_plus_b2 = Bunch(a=4, b='bee', mergedint=7, mergedstr='b1b2', c='see')
   assert (b1 + b2) == b1_plus_b2

def test_bunch_visit():
   count = 0

   def func(k, v, depth):
      # print('    ' * depth, k, type(v))
      nonlocal count
      count += 1
      if v == 'b': return True
      return False

   b = Bunch(a='a', b='b', bnch=Bunch(foo='bar'))
   b.visit_remove_if(func)
   assert b == Bunch(a='a', bnch=Bunch(foo='bar'))
   assert count == 4

if __name__ == "__main__":
   from tempfile import mkdtemp

   test_bunch_pickle(mkdtemp())
   test_bunch_init()
   test_bunch_sub()
   test_bunch_items()
   test_bunch_add()
   test_bunch_visit()