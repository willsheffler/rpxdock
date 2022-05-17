import rpxdock as rp, concurrent, pytest, numpy as np
from willutil import Bunch

def test_P6_grid_onecomp_notrim(hscore, bodyC3):
   kw = rp.app.defaults()
   kw.wts = Bunch(ncontact=0.01, rpx=1.0)
   kw.beam_size = 2e4
   kw.max_bb_redundancy = 2.0
   kw.max_delta_h = 9999
   kw.nout_debug = 0
   kw.nout_top = 0
   kw.nout_each = 0
   kw.score_only_ss = 'H'
   kw.max_trim = 0
   kw.flip_components = [True]

   # kw.debug = True
   kw.executor = concurrent.futures.ThreadPoolExecutor(min(4, kw.ncpu / 2))

   spec = Bunch(
      nfold=6,
      axis=[0, 0, 1],
      cartaxis=[1, 0, 0],
   )

   # spec = rp.search.DockSpec1CompCage('T3')
   # sampler = rp.sampling.hier_axis_sampler(spec.nfold, lb=0, ub=100, resl=10, angresl=10,  axis=spec.axis,
   sampler = rp.sampling.grid_sym_axis(
      cart=np.arange(0, 10, 5),
      ang=np.arange(0, 360 / spec.nfold, 5),
      axis=spec.axis,
      cartaxis=spec.cartaxis,
      flip=None,
   )

   result = rp.search.make_onecomp_layer(
      bodyC3,
      spec,
      hscore,
      rp.grid_search,
      sampler,
      **kw,
   )
   # print(result)
   result.dump_pdbs_top_score(
      hscore=hscore,
      **kw.sub(nout_top=10, output_prefix='test_cage_hier_onecomp_notrim'),
   )
   #
   # rp.dump(result, 'rpxdock/data/testdata/test_p6_grid_onecomp_notrim.pickle')
   # ref = rp.data.get_test_data('test_p6_grid_onecomp_notrim')

def main():
   hscore = rp.data.small_hscore()
   # hscore = rp.RpxHier('ilv_h/1000', hscore_data_dir='/home/sheffler/data/rpx/hscore')
   # C2 = rp.data.get_body('C2_REFS10_1')
   C3 = rp.data.get_body('C3_1na0-1_1')

   test_P6_grid_onecomp_notrim(hscore, C3)

   # test_deepesh_1comp_bug(hscore)

if __name__ == '__main__':
   main()
