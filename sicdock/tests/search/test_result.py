import _pickle, collections
from sicdock.search.result import *

def test_result(result):
   a = result.copy()
   a.attrs['foo'] = 1
   b = result.copy()
   b.attrs['bar'] = 2
   c = concat_results([a, b])
   assert len(c.model) == len(a.model) + len(b.model)
   assert c.meta == [a.attrs, b.attrs]
   assert np.all(c.imeta == np.repeat([0, 1], len(result)))

def test_result_pickle(result, tmpdir):
   with open(tmpdir + '/foo', 'wb') as out:
      _pickle.dump(result, out)
   with open(tmpdir + '/foo', 'rb') as inp:
      result2 = _pickle.load(inp)
   assert result == result2

def test_result_attrs():
   result = Result(foo=np.array(10), attrs=dict(bar='baz'))
   assert result.bar == 'baz'

def test_mismatch_len(result):
   result2 = dummy_result(100)
   r = concat_results([result, result2])
   assert len(result) != len(result2)
   assert len(r) == len(result) + len(result2)

if __name__ == '__main__':
   import tempfile
   test_result(dummy_result(1000))
   test_result_pickle(dummy_result(1000), tempfile.mkdtemp())
   test_result_attrs()
   test_mismatch_len(dummy_result(1000))
