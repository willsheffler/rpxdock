import rpxdock as rp, numpy as np, pytest
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor

def testarg():
   kw = rp.app.defaults()
   kw.wts = rp.Bunch(ncontact=0.01, rpx=1.0)
   kw.beam_size = 1e4
   kw.max_bb_redundancy = 3.0
   kw.max_longaxis_dot_z = 0.5
   kw.executor = ThreadPoolExecutor(min(4, kw.ncpu / 2))
   kw.multi_iface_summary = np.min
   kw.debug = True
   return kw

def test_asym(hscore, body, body2):
   kw = testarg()
   kw.max_trim = 0
   kw.output_prefix = 'test_asym'

   cartlb = np.array([+00, +10, +00])
   cartub = np.array([+30, +20, +30])
   cartbs = np.array([4, 1, 4], dtype="i")
   sampler = rp.sampling.XformHier_f4(cartlb, cartub, cartbs, 30)

   # sampler = rp.search.asym_get_sample_hierarchy(body2, hscore, 18)
   # print(f'toplevel samples {sampler.size(0):,}')
   result = rp.search.make_asym([body2, body], hscore, sampler, **kw)

   # result.dump_pdbs_top_score(10, hscore=hscore, wts=kw.wts)

   # rp.dump(result, 'rpxdock/data/testdata/test_asym.pickle')
   ref = rp.data.get_test_data('test_asym')
   rp.search.assert_results_close(result, ref)

@pytest.mark.skip
def test_asym_trim(hscore, body, body2):
   kw = testarg()
   kw.max_trim = 100
   kw.output_prefix = 'test_asym_trim'

   kw.beam_size = 2e5
   kw.executor = None

   cartlb = np.array([-40, +00, -40])
   cartub = np.array([+40, +40, +40])
   cartbs = np.array([12, 12, 12], dtype="i")
   sampler = rp.sampling.XformHier_f4(cartlb, cartub, cartbs, 30)

   # sampler = rp.search.asym_get_sample_hierarchy(body2, hscore, 18)
   # print(f'toplevel samples {sampler.size(0):,}')
   result = rp.search.make_asym([body2, body], hscore, sampler, **kw)

   result.dump_pdbs_top_score(10, hscore=hscore, wts=kw.wts)

   rp.dump(result, 'rpxdock/data/testdata/test_asym_trim.pickle')
   ref = rp.data.get_test_data('test_asym_trim')
   rp.search.assert_results_close(result, ref)

def main():
   hscore = rp.data.small_hscore()
   # hscore = rp.RpxHier('ilv_h/1000', hscore_data_dir='/home/sheffler/data/rpx/hscore')
   # hscore = rp.RpxHier('ilv_h', hscore_data_dir='/home/sheffler/data/rpx/hscore')
   # hscore.score_only_sspair = ['HH']

   body1 = rp.data.get_body('DHR14')
   body2 = rp.data.get_body('top7')
   # body1.score_only_ss = 'H'
   # body2.score_only_ss = 'H'
   # body1 = rp.Body('rpxdock/data/pdb/DHR14.pdb.gz')
   # body2 = rp.Body('rpxdock/data/pdb/top7.pdb.gz')

   # body1 = rp.data.get_body('top7b')

   test_asym(hscore, body1, body2)
   # test_asym_trim(hscore, body1, body2)

if __name__ == '__main__':
   main()