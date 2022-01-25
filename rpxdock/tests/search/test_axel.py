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

def test_axel(hscore):
   C3_A = rp.data.get_body_copy('C3_1na0-1_1')
   C3_B = rp.data.get_body_copy('C3_1nza_1')
   C3_A.init_coords(3, [0, 0, 1, 0])
   C3_B.init_coords(3, [0, 0, 1, 0])

   kw = get_arg(fixed_components=True)
   kw.beam_size = 5000
   spec = rp.search.DockSpecAxel("AXEL_3")

   sampler1 = rp.sampling.hier_axis_sampler(3, 0, 100, 10, 10, axis=spec.axis[0],
                                            flipax=spec.flip_axis[0])
   sampler2 = rp.sampling.ZeroDHier([np.eye(4), rp.homog.hrot([1, 0, 0], 180)])
   sampler = rp.sampling.CompoundHier(sampler2, sampler1)
   _, x = sampler.get_xforms(resl=0, idx=[0])

   result = rp.search.make_multicomp([C3_A, C3_B], spec, hscore, rp.hier_search, sampler, **kw)
   print(result)

#   result.dump_pdbs_top_score(hscore=hscore,
#                               **kw.sub(nout_top=10, output_prefix='test_axel'))

def test_axel_asym(hscore):
   C2_A = rp.data.get_body_copy('C2_3hm4_1')
   C2_A.init_coords(2, [0, 0, 1, 0])
   C3_B = rp.data.get_body_copy('C3_1nza_1')
   C3_B.init_coords(3, [0, 0, 1, 0])

   kw = get_arg(fixed_components=True)
   kw.beam_size = 5000
   spec = rp.search.DockSpecAxel("AXEL_1_2_3")

   sampler1 = rp.sampling.hier_axis_sampler(2 * 3, 0, 100, 10, 10, axis=spec.axis[0],
                                            flipax=spec.flip_axis[0])
   sampler2 = rp.sampling.ZeroDHier([np.eye(4), rp.homog.hrot([1, 0, 0], 180)])
   sampler = rp.sampling.CompoundHier(sampler2, sampler1)
   _, x = sampler.get_xforms(resl=0, idx=[0])

   result = rp.search.make_multicomp([C2_A, C3_B], spec, hscore, rp.hier_search, sampler, **kw)
   print(result)

#   result.dump_pdbs_top_score(hscore=hscore,
#                               **kw.sub(nout_top=50, output_prefix='test_axel_asym'))

def test_axel_grid(hscore):
   C3_A = rp.data.get_body_copy('C3_1na0-1_1')
   C3_A.init_coords(3, [0, 0, 1, 0])
   C3_B = rp.data.get_body_copy('C3_1nza_1')
   C3_B.init_coords(3, [0, 0, 1, 0])

   kw = get_arg(fixed_components=True)
   kw.beam_size = 5000
   spec = rp.search.DockSpecAxel("AXEL_3")

   sampler1 = rp.sampling.grid_sym_axis(
      cart=np.arange(0, 100, 2),
      ang=np.arange(0, 360 / 3, 2),
      axis=spec.axis[0],
      flip=list(spec.flip_axis[0])[:3],
   )
   shape = (2, 2 * len(sampler1), 4, 4)
   sampler = np.zeros(shape=shape)
   sampler[0, :len(sampler1)] = sampler1
   sampler[0, len(sampler1):] = sampler1
   sampler[1, :len(sampler1)] = np.eye(4)
   sampler[1, len(sampler1):] = rp.homog.hrot([1, 0, 0], 180)
   sampler = np.swapaxes(sampler, 0, 1)

   result = rp.search.make_multicomp([C3_A, C3_B], spec, hscore, rp.grid_search, sampler, **kw)
   print(result)

#   result.dump_pdbs_top_score(hscore=hscore,
#                               **kw.sub(nout_top=10, output_prefix='test_axel_grid'))

if __name__ == "__main__":
   #logging.getLogger().setLevel(level='INFO')

   hscore = rp.data.small_hscore()

   # C3_A = rp.data.get_body_copy('C3_1na0-1_1')
   # C3_B = rp.data.get_body_copy('C3_1nza_1')
   # C3_A.init_coords(3, [0, 0, 1, 0])
   # C3_B.init_coords(3, [0, 0, 1, 0])
   # C2_A = rp.data.get_body_copy('C2_3hm4_1')
   # C2_A.init_coords(2, [0, 0, 1, 0])

   test_axel(hscore)
   test_axel_asym(hscore)
   test_axel_grid(hscore)
