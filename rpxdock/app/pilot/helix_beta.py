import rpxdock as rp, numpy as np, argparse

def dock_helix(hscore, body, **arg):
   arg = rp.Bunch(arg)
   assert arg.max_trim == 0, 'no support for trimming yet'

   # arg.executor = ThreadPoolExecutor(8)
   arg.executor = None

   print(arg.cart_bounds)
   assert len(arg.cart_bounds) is 3, 'improper cart_bounds'
   cartlb = np.array([arg.cart_bounds[0][0], arg.cart_bounds[1][0], arg.cart_bounds[2][0]])
   cartub = np.array([arg.cart_bounds[0][1], arg.cart_bounds[1][1], arg.cart_bounds[2][1]])
   cartbs = np.ceil((cartub - cartlb) / arg.cart_resl).astype('i')
   print('cart lower bound', cartlb)
   print('cart upper bound', cartub)
   print('cart base block nside', cartbs)
   sampler = rp.sampling.XformHier_f4(cartlb, cartub, cartbs, arg.ori_resl)
   # sampler = rp.search.asym_get_sample_hierarchy(body2, hscore, 18)

   print(f'toplevel samples {sampler.size(0):,}')
   result = rp.search.make_helix(body, hscore, sampler, **arg)
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
   parser.add_argument("--cart_cell_width", type=float, default=10,
                       help='cartesian resolution of the initial search stage, default 10')
   parser.add_argument("--angle_cell_width", type=float, default=30,
                       help='angular resolution of the initial search stage, default 30')
   arg = rp.options.get_cli_args(parent=parser, dont_set_default_cart_bounds=True)
   if not arg.cart_bounds:
      arg.cart_bounds = np.array([(0, 100), (-100, 100), (-100, 100)])
   else:
      arg.cart_bounds = rp.options.process_cart_bounds(arg.cart_bounds)

   arg.iresl_second_shift = 2
   arg.helix_min_primary_angle = 360 / arg.helix_max_isecond - 1
   arg.helix_max_primary_angle = 360 / arg.helix_min_isecond + 1
   arg.max_iclash = int(arg.helix_max_isecond * 1.2 + 3)

   arg.symframe_num_helix_repeats = arg.helix_max_isecond * 2 + 2

   return arg

def main():
   arg = get_helix_args()
   hscore = rp.CachedProxy(rp.RpxHier(arg.hscore_files, **arg))

   # two extra sampling refinements
   arg.nresl = hscore.actual_nresl + arg.helix_iresl_second_shift

   bodies = [rp.Body(inp, **arg) for inp in arg.inputs1]

   results = list()
   for body in bodies:
      result = dock_helix(hscore, body, **arg)
      results.append(result)
   result = rp.concat_results(results)
   print(result)
   if arg.dump_pdbs:
      result.dump_pdbs_top_score(score=hscore, **arg)
      result.dump_pdbs_top_score_each(hscore=hscore, **arg)
   if not arg.suppress_dump_results:
      rp.util.dump(result, arg.output_prefix + '_Result.pickle')

if __name__ == '__main__':
   main()