import concurrent, os, argparse, sys, numpy as np, rpxdock as rp, pytest

def get_arg(**kw):
   arg = rp.app.defaults()
   arg.wts = rp.Bunch(ncontact=0.3, rpx=1.0)
   arg.beam_size = 2e4
   arg.max_bb_redundancy = 2.0
   arg.max_delta_h = 9999
   arg.nout_debug = 0
   arg.nout_top = 0
   arg.nout_each = 0
   arg.score_only_ss = 'H'
   arg.max_trim = 0
   # arg.debug = True
   arg.executor = concurrent.futures.ThreadPoolExecutor(min(4, arg.ncpu / 2))
   return arg.sub(kw)

def test_sic_axel(hscore, C3_A, C3_B):
   kw = get_arg(fixed_components=True)
   kw.beam_size = 5000

   sampler1 = rp.sampling.hier_axis_sampler(3, 0, 100, 10, 10)
   sampler2 = rp.sampling.ZeroDHier([np.eye(4),rp.homog.hrot([1,0,0],180)])
 #  sampler2 = rp.sampling.ZeroDHier(np.eye(4))
   sampler = rp.sampling.CompoundHier(sampler2, sampler1)
   _,x = sampler.get_xforms(resl=0, idx=[0])
#   print(x)
#   print(x.shape)
#   assert 0

   dummy = np.eye(4)
   dummy[:3,3] = [10000,0,0]
   spec = rp.Bunch(nfold=[3,3], axis=[[0,0,1,0],[0,0,1,0]], arch="axel_1", num_components=2,
           to_neighbor_olig=[dummy,dummy])


   result = rp.search.make_multicomp([C2_A, C3_B], spec, hscore, rp.hier_search,
                                     sampler, **kw)
   print(result)
   result.dump_pdbs_top_score(hscore=hscore,
                               **kw.sub(nout_top=10, output_prefix='test_sic_axel'))

def test_sic_axel_asym(hscore, C2_A, C3_B):
   kw = get_arg(fixed_components=True)
   kw.beam_size = 5000

   sampler1 = rp.sampling.hier_axis_sampler(3, 0, 100, 10, 10)
   sampler2 = rp.sampling.ZeroDHier([np.eye(4),rp.homog.hrot([1,0,0],180)])
 #  sampler2 = rp.sampling.ZeroDHier(np.eye(4))
   sampler = rp.sampling.CompoundHier(sampler2, sampler1)
   _,x = sampler.get_xforms(resl=0, idx=[0])
#   print(x)
#   print(x.shape)
#   assert 0

   dummy = np.eye(4)
   dummy[:3,3] = [10000,0,0]
#   spec = rp.Bunch(nfold=[3,3], axis=[[0,0,1,0],[0,0,1,0]], arch="axel_1", num_components=2,
#           to_neighbor_olig=[dummy,dummy])
   spec = rp.Bunch(nfold=[2,3], axis=[[0,0,1,0],[0,0,1,0]], arch="axel_1", num_components=2,
           to_neighbor_olig=[dummy,dummy])


   result = rp.search.make_multicomp([C2_A, C3_B], spec, hscore, rp.hier_search,
                                     sampler, **kw)
   print(result)
   result.dump_pdbs_top_score(hscore=hscore,
                               **kw.sub(nout_top=10, output_prefix='test_sic_axel'))

if __name__=="__main__":
   #logging.getLogger().setLevel(level='INFO')

   hscore = rp.data.small_hscore()

   C3_A = rp.data.get_body('C3_1na0-1_1')
   C3_B = rp.data.get_body('C3_1nza_1')
   C3_A.init_coords(3, [0,0,1,0])
   C3_B.init_coords(3, [0,0,1,0])
   C2_A = rp.data.get_body('C2_3hm4_1')
   C2_A.init_coords(2, [0,0,1,0])

   test_sic_axel(hscore, C3_A, C3_B)
   test_sic_axel_asym(hscore, C2_A, C3_B)
