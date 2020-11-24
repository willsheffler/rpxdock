import sys, rpxdock as rp, numpy as np, argparse

def dock_helix(hscore, body, **kw):
   kw = rp.Bunch(kw)

   print(f'{"helix_beta.py:dock_helix starting":=^80}')

   assert kw.max_trim == 0, 'no support for trimming yet'

   # kw.executor = ThreadPoolExecutor(8)
   kw.executor = None

   assert len(kw.cart_bounds) is 3, 'improper cart_bounds'
   cartlb = np.array([kw.cart_bounds[0][0], kw.cart_bounds[1][0], kw.cart_bounds[2][0]])
   cartub = np.array([kw.cart_bounds[0][1], kw.cart_bounds[1][1], kw.cart_bounds[2][1]])
   cartbs = np.ceil((cartub - cartlb) / kw.cart_resl).astype('i')
   print('cart lower bound', cartlb)
   print('cart upper bound', cartub)
   print('cart base block nside', cartbs)
   sampler = rp.sampling.XformHier_f4(cartlb, cartub, cartbs, kw.ori_resl)
   # sampler = rp.search.asym_get_sample_hierarchy(body2, hscore, 18)

   print(f'toplevel samples {sampler.size(0):,}')
   result = rp.search.make_helix(body, hscore, sampler, **kw)
   return result

def get_helix_args():
   parser = argparse.ArgumentParser(allow_abbrev=False)
   parser.add_argument("--helix_min_isecond", type=int, default=8,
                       help='minimum number of steps to second interface, default 8')
   parser.add_argument("--helix_max_isecond", type=int, default=16,
                       help='maximum number of steps to second interface, default 16')
   parser.add_argument("--helix_iresl_second_shift", type=int, default=2,
                       help='steps decreased resolution of second inferface, default 2')
   parser.add_argument(
      "--helix_min_primary_score", type=float, default=30,
      help='minimum score for the primary interface, mainly for speed, default=30')
   parser.add_argument("--helix_max_primary_score", type=float, default=100,
                       help='maximum score for the primary interface, default 100')
   parser.add_argument("--helix_min_delta_z", type=float, default=0.001,
                       help='minimum shift alone helix axis per step, default 0.001')
   parser.add_argument("--helix_max_delta_z", type=float, default=None,
                       help='maximum shift alone helix axis per step, default uses body radius')
   parser.add_argument(
      "--helix_min_primary_angle", type=float, default=None,
      help='minimum rotation around axis per step. default based on max_isecond is usually the right choice'
   )
   parser.add_argument(
      "--helix_max_primary_angle", type=float, default=None,
      help='maximum rotation around axis per step. default based on min_isecond is usually the right choice'
   )
   # parser.add_argument("--cart_cell_width", type=float, default=10,
   #                     help='cartesian resolution of the initial search stage, default 10')
   # parser.add_argument("--angle_cell_width", type=float, default=30,
   #                     help='angular resolution of the initial search stage, default 30')

   parser.add_argument("--tether_xform", type=str, nargs='*', default=[],
                       help='two structs with configuration to sample around')
   parser.add_argument("--tether_dist", type=float, default=3,
                       help='max dist from supplied dimer configuration')
   parser.add_argument("--tether_ang", type=float, default=10,
                       help='max angle from supplied dimer configuration')

   parser.add_argument("--helix_min_radius", type=float, default=0, help='min helix radius')
   parser.add_argument("--helix_max_radius", type=float, default=9e9, help='max helix radius')

   parser.add_argument("--tmpa", type=int)
   parser.add_argument("--tmpb", type=int)

   kw = rp.options.get_cli_args(parent=parser, dont_set_default_cart_bounds=True)

   if not kw.cart_bounds:
      kw.cart_bounds = np.array([(0, 100), (-100, 100), (-100, 100)])
   else:
      kw.cart_bounds = rp.options._process_cart_bounds(kw.cart_bounds)

   kw.iresl_second_shift = 2
   kw.helix_min_primary_angle = 360 / kw.helix_max_isecond - 1
   kw.helix_max_primary_angle = 360 / kw.helix_min_isecond + 1
   kw.max_iclash = int(kw.helix_max_isecond * 1.2 + 3)

   kw.symframe_num_helix_repeats = kw.helix_max_isecond * 2 + 2

   if not kw.inputs1:
      print('No inputs! Aborting.')
      sys.exit(-1)

   rp.options.print_options(kw)

   return kw

def main():
   kw = get_helix_args()
   hscore = rp.CachedProxy(rp.RpxHier(kw.hscore_files, **kw))

   # two extra sampling refinements
   kw.nresl = hscore.actual_nresl + kw.helix_iresl_second_shift

   bodies = [rp.Body(inp, **kw) for inp in kw.inputs1]

   results = list()
   for body in bodies:
      result = dock_helix(hscore, body, **kw)
      results.append(result)
   result = rp.concat_results(results)
   print(result)
   if kw.dump_pdbs:
      result.dump_pdbs_top_score(score=hscore, **kw)
      result.dump_pdbs_top_score_each(hscore=hscore, **kw)
   if not kw.suppress_dump_results:
      rp.util.dump(result, kw.output_prefix + '_Result.pickle')

if __name__ == '__main__':
   main()
