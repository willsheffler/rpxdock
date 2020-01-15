import rpxdock as rp, numpy as np, pytest
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor

def testarg():
   arg = rp.app.defaults()
   arg.wts = rp.Bunch(ncontact=0.01, rpx=1.0)
   arg.beam_size = 1e4
   arg.max_bb_redundancy = 3.0
   arg.max_longaxis_dot_z = 0.5
   arg.executor = ThreadPoolExecutor(min(4, arg.ncpu / 2))
   arg.multi_iface_summary = np.min
   arg.debug = True
   return arg

@pytest.mark.skip
def test_helix(hscore, body_tiny):
   arg = testarg()
   arg.max_trim = 0
   arg.output_prefix = 'test_helix'
   arg.beam_size = 1e5
   arg.max_bb_redundancy = 3.0
   arg.wts.ncontact = 0

   harg = rp.Bunch()
   harg.min_isecond = 8
   harg.max_isecond = 16
   harg.min_primary_score = 30
   harg.max_primary_score = 100
   harg.iresl_second_shift = 2
   harg.min_delta_z = 0.001
   harg.max_delta_z = body_tiny.radius_max() * 2 / harg.min_isecond
   harg.min_primary_angle = 360 / harg.max_isecond - 1
   harg.max_primary_angle = 360 / harg.min_isecond + 1
   harg.max_iclash = int(harg.max_isecond * 1.2 + 3)
   arg.helix_args = harg

   # two extra sampling refinements
   arg.nresl = hscore.actual_nresl + harg.iresl_second_shift
   arg.symframe_num_helix_repeats = harg.max_isecond * 2 + 2

   # arg.executor = ThreadPoolExecutor(8)
   arg.executor = None

   # cartlb = np.array([00, -150, -150])
   # cartub = np.array([150, 150, 150])
   # cartbs = np.array([15, 30, 30], dtype="i")
   cartlb = np.array([00, -100, -100])
   cartub = np.array([100, 100, 100])
   cartbs = np.array([10, 20, 20], dtype="i")
   sampler = rp.sampling.XformHier_f4(cartlb, cartub, cartbs, 30)
   # sampler = rp.search.asym_get_sample_hierarchy(body2, hscore, 18)

   print(f'toplevel samples {sampler.size(0):,}')
   result = rp.search.make_helix(body_tiny, hscore, sampler, **arg)
   print(result)
   result.dump_pdbs_top_score(hscore=hscore, **arg)

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