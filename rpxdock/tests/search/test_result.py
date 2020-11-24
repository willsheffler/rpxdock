import _pickle, collections, pytest
from rpxdock.search.result import *

def test_result(result):
   a = result.copy()
   a.attrs['foo'] = 1
   b = result.copy()
   b.attrs['bar'] = 2
   c = concat_results([a, b])
   assert len(c.model) == len(a.model) + len(b.model)
   assert c.dockinfo == [a.attrs, b.attrs]
   assert np.all(c.ijob == np.repeat([0, 1], len(result)))

def test_result_pickle(result, tmpdir):
   with open(tmpdir + '/foo', 'wb') as out:
      _pickle.dump(result, out)
   with open(tmpdir + '/foo', 'rb') as inp:
      result2 = _pickle.load(inp)
   assert result == result2
   assert isinstance(result, Result)
   assert isinstance(result2, Result)

def test_result_attrs():
   result = Result(foo=np.array(10), attrs=dict(bar='baz'))
   assert result.bar == 'baz'

def test_mismatch_len(result):
   result2 = dummy_result(100)
   r = concat_results([result, result2])
   assert len(result) != len(result2)
   assert len(r) == len(result) + len(result2)

def test_top_each(result):
   n = 13
   top = result.top_each(n)
   for k, v in top.items():
      assert len(v) == 13
      w = np.where(result.ijob == k)[0]
      s = result.scores[w]
      o = np.argsort(-s)[:n]
      assert np.allclose(s[o], result.scores[v])

def test_result_no_body_label(result):
   foo = Result(result.data, body_=['a', 'b', 'c'])
   assert foo.body_label_ == 'body0 body1 body2'.split()

@pytest.mark.skip
def test_result_coords():
   r = rp.data.get_test_data('test_cage_hier_no_trim')
   r = rp.concat_results([r])
   idx = 7
   bodyA, bodyB = r.bodies[r.ijob[idx].data]
   crdA = bodyA.positioned_coord(pos=r.xforms[idx, 0])
   crdB = bodyB.positioned_coord(pos=r.xforms[idx, 1])

   assert 0

if __name__ == '__main__':
   import tempfile
   # test_result(dummy_result(1000))
   # test_result_pickle(dummy_result(1000), tempfile.mkdtemp())
   # test_result_attrs()
   # test_mismatch_len(dummy_result(1000))
   # test_top_each(dummy_result(1000))
   # test_result_no_body_label(dummy_result(1000))
   test_result_coords()
