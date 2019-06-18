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

if __name__ == "__main__":
   from tempfile import mkdtemp

   test_bunch_pickle(mkdtemp())
   test_bunch_init()
   test_bunch_sub()
   test_bunch_items()
