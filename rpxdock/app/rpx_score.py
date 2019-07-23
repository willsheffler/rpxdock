#! /home/sheffler/.conda/envs/rpxdock/bin/python

import logging, os, rpxdock as rp, numpy as np

def get_opts():
   parser = rp.options.default_cli_parser()
   addarg = rp.options.add_argument_unless_exists(parser)
   addarg("--bodies", type=str, nargs='+', required=True)
   addarg("--other_bodies", type=str, nargs='+', default=[])
   addarg("--dump_scored_pdb", action='store_true', default=False)
   arg = parser.parse_args()
   return rp.options.process_cli_args(arg)

def score_onebody(hscore, **kw):
   arg = rp.Bunch(kw)
   for fn in arg.bodies:
      body = rp.Body(fn)
      iscores = hscore.score_matrix_intra(body, arg.wts)
      overall = np.sum(np.max(iscores, axis=0))
      # print(np.max(iscores, axis=1))
      meanmean = np.mean(iscores)
      meanmax = np.mean(np.max(iscores, axis=0))
      meanmax = np.max(np.mean(iscores, axis=0))
      maxmax = np.max(iscores)
      summean = np.sum(np.mean(iscores, axis=0))
      summax = np.sum(np.max(iscores, axis=0))
      sumsum = np.sum(iscores)
      bfac_file = ''
      if arg.dump_scored_pdb:
         resscore = np.sum(iscores, axis=0)
         bfac_file = os.path.basename(fn) + '_rpx.pdb'
         body.dump_pdb(bfac_file, bfactor=resscore, use_orig_coords=True)
      print(f'{fn} sum {sumsum:7.2f} mean {meanmean:7.3f} sum(max) {summax:8.3f} ' +
            f'sum(mean) {summean:7.3f} {bfac_file}')

def score_twobody(hscore, **kw):
   arg = rp.Bunch(kw)
   for fn1 in arg.bodies:
      body1 = rp.Body(fn1)
      for fn2 in arg.other_bodies:
         body2 = Body(fn2)
         print(fn1, fn2, hscore.score(body1, body2, arg.wts))

def main():
   arg = get_opts()
   logging.info(f'weights: {arg.wts}')

   hscore = rp.HierScore(arg.hscore_files, **arg)

   if arg.other_bodies:
      score_twobody(hscore, **arg)
   else:
      score_onebody(hscore, **arg)

if __name__ == '__main__':
   main()