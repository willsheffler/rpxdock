import rpxdock as rp, numpy as np, pytest

def testarg():
   kw = rp.app.defaults()
   kw.wts = rp.Bunch(ncontact=0.01, rpx=1.0)
   kw.beam_size = 1e4
   kw.max_bb_redundancy = 3.0
   kw.max_longaxis_dot_z = 0.5
   # kw.executor = ThreadPoolExecutor(min(4, kw.ncpu / 2))
   kw.multi_iface_summary = np.min
   kw.debug = True
   return kw

@pytest.mark.skip
def test_helix(hscore, body_tiny):
   kw = testarg()
   kw.max_trim = 0
   kw.output_prefix = 'test_helix'
   kw.beam_size = 1e5
   kw.max_bb_redundancy = 3.0
   kw.wts.ncontact = 0

   kw.helix_min_isecond = 8
   kw.helix_max_isecond = 16
   kw.helix_min_primary_score = 30
   kw.helix_max_primary_score = 100
   kw.helix_iresl_second_shift = 2
   kw.helix_min_delta_z = 0.001
   kw.helix_max_delta_z = body_tiny.radius_max() * 2 / kw.helix_min_isecond
   kw.helix_min_primary_angle = 360 / kw.helix_max_isecond - 1
   kw.helix_max_primary_angle = 360 / kw.helix_min_isecond + 1
   kw.helix_max_iclash = int(kw.helix_max_isecond * 1.2 + 3)

   # two extra sampling refinements
   kw.nresl = hscore.actual_nresl + kw.helix_iresl_second_shift
   kw.symframe_num_helix_repeats = kw.helix_max_isecond * 2 + 2

   # cartlb = np.array([00, -150, -150])
   # cartub = np.array([150, 150, 150])
   # cartbs = np.array([15, 30, 30], dtype="i")
   cartlb = np.array([+000, -100, -100])
   cartub = np.array([+100, +100, +100])
   cartbs = np.array([10, 20, 20], dtype="i")
   sampler = rp.sampling.XformHier_f4(cartlb, cartub, cartbs, 30)
   # sampler = rp.search.asym_get_sample_hierarchy(body2, hscore, 18)

   print(f'toplevel samples {sampler.size(0):,}')
   result = rp.search.make_helix(body_tiny, hscore, sampler, **kw)
   print(result)
   result.dump_pdbs_top_score(hscore=hscore, **kw)

   # rp.dump(result, 'rpxdock/data/testdata/test_asym.pickle')
   # ref = rp.data.get_test_data('test_asym')
   # rp.search.assert_results_close(result, ref)

def main():
   import logging
   from logging import getLogger, getLevelName, Formatter, StreamHandler

   log = getLogger()
   log.setLevel(getLevelName('INFO'))
   log_formatter = Formatter("%(name)s: %(message)s [%(threadName)s] ")
   console_handler = StreamHandler()
   console_handler.setFormatter(log_formatter)
   log.addHandler(console_handler)

   # hscore = rp.data.small_hscore()
   hscore = rp.RpxHier('ilv_h/1000', hscore_data_dir='/home/sheffler/data/rpx/hscore')
   # hscore = rp.RpxHier('ilv_h', hscore_data_dir='/home/sheffler/data/rpx/hscore')
   # hscore = rp.RpxHier('afilmv_elh', hscore_data_dir='/home/sheffler/data/rpx/hscore')
   # body = rp.data.get_body('tiny')
   # body = rp.data.get_body('top7')
   # body = rp.data.get_body('C3_1na0-1_1')
   body = rp.data.get_body('DHR14')
   test_helix(hscore, body)

if __name__ == '__main__':
   main()