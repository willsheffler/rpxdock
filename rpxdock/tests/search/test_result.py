import _pickle, collections, pytest, rpxdock as rp, tempfile
from rpxdock.search.result import *
import willutil as wu

def main():
   import tempfile
   # test_result(dummy_result(1000))
   # test_result_pickle(dummy_result(1000), tempfile.mkdtemp())
   # test_result_attrs()
   # test_mismatch_len(dummy_result(1000))
   # test_top_each(dummy_result(1000))
   test_result_no_body_label(dummy_result(1000))
   # test_result_coords()
   test_result_tarball()

   # import tempfile
   # # test_result(dummy_result(1000))
   # test_result_pickle(dummy_result(1000), tempfile.mkdtemp())
   # # test_result_attrs()
   # # test_mismatch_len(dummy_result(1000))
   # test_top_each(dummy_ressult(1000))
   # test_result_no_body_label(dummy_result(1000))
   # test_result_dump_asym()

def test_result_tarball():
   # timer = wu.Timer()
   with tempfile.TemporaryDirectory() as tmpdir:

      r = rp.data.get_test_data('test_result.pickle')

      outfiles = r.dump_pdbs_top_score(1, output_prefix=f'{tmpdir}/')

      f = os.listdir(tmpdir)
      assert len(f) == 1
      with open(f'{tmpdir}/{f[0]}') as inp:
         pdb1 = inp.read()

      result_to_tarball(r, f'{tmpdir}/test.result.txz', overwrite=True)

      r2 = result_from_tarball(f'{tmpdir}/test.result.txz')

      rp.search.result.assert_results_close(r, r2)
      # assert r.body_label_ == r2.body_label_
      assert r.pdb_extra_ == r2.pdb_extra_

      len(r.bodies) == len(r2.bodies)
      for i, (l1, l2) in enumerate(zip(r.bodies, r2.bodies)):
         for j, (b1, b2) in enumerate(zip(l1, l2)):
            assert np.allclose(b1.coord, b2.coord, atol=1e-1)
            assert np.allclose(b1.stub, b2.stub, atol=1e-3)
            assert str(b1.ss) == str(b2.ss)
            assert str(b1.seq) == str(b2.seq)

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
   foo = Result(result.data, body_=[None, None, None])
   assert foo.body_label_ == ['body_0_0 body_0_1 body_0_2'.split()]

# @pytest.mark.skip('no test criterion')
# def test_result_dump_asym():
#    # assert 0, 'fix dump_pdb output_asym_only'
#    result = rp.data.get_test_data('result_test_asym_out')
#    print(result.dump_pdbs(output_asym_only=True))

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
   main()
