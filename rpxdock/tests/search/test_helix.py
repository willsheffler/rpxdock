import rpxdock as rp
from willutil import Bunch
import numpy as np, pytest

def _test_args():
   kw = rp.app.defaults()
   kw.wts = Bunch(ncontact=0.01, rpx=1.0)
   kw.beam_size = 1e4
   kw.max_bb_redundancy = 3.0
   kw.max_longaxis_dot_z = 0.5
   # kw.executor = ThreadPoolExecutor(min(4, kw.ncpu / 2))
   kw.multi_iface_summary = np.min
   kw.debug = True
   return kw

def test_helix(hscore, body_tiny):
   kw = _test_args()
   kw.max_trim = 0
   kw.output_prefix = 'test_helix'
   kw.beam_size = 10_000
   kw.max_bb_redundancy = 3.0
   kw.wts.ncontact = 0

   kw.helix_min_isecond = 8
   kw.helix_max_isecond = 16
   kw.helix_min_radius = 0
   kw.helix_max_radius = 9e9
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
   cartlb = np.array([+0, -0, -0])
   cartub = np.array([+120, +90, +90])
   cartbs = np.array([4, 3, 3], dtype="i")
   sampler = rp.sampling.XformHier_f4(cartlb, cartub, cartbs, 40)
   # sampler = rp.search.asym_get_sample_hierarchy(body2, hscore, 18)

   print(f'toplevel samples {sampler.size(0):,}')
   result = rp.search.make_helix(body_tiny, hscore, sampler, **kw)
   # result.dump_pdbs_top_score(hscore=hscore, **kw)

   rp.search.result_to_tarball(result, 'rpxdock/data/testdata/test_helix.result.txz',
                               overwrite=True)
   ref = rp.data.get_test_data('test_helix')

   print('-' * 80)
   print(result)
   print('-' * 80)
   print(ref)
   print('-' * 80)

   rp.search.assert_results_close(result, ref)

def main():
   import logging
   from logging import getLogger, getLevelName, Formatter, StreamHandler

   log = getLogger()
   log.setLevel(getLevelName('INFO'))
   log_formatter = Formatter("%(name)s: %(message)s [%(threadName)s] ")
   console_handler = StreamHandler()
   console_handler.setFormatter(log_formatter)
   log.addHandler(console_handler)

   hscore = rp.data.small_hscore()
   # hscore = rp.RpxHier('ilv_h/1000',
   # hscore_data_dir='/home/sheffler/data/rpx/hscore/willsheffler')
   # hscore = rp.RpxHier('ilv_h', hscore_data_dir='/home/sheffler/data/rpx/hscore')
   # hscore = rp.RpxHier('afilmv_elh', hscore_data_dir='/home/sheffler/data/rpx/hscore')
   # body = rp.data.get_body('tiny')
   # body = rp.data.get_body('top7')
   # body = rp.data.get_body('C3_1na0-1_1')
   body = rp.data.get_body('DHR14')
   test_helix(hscore, body)

if __name__ == '__main__':
   main()