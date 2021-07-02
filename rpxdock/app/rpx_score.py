#! /home/sheffler/.conda/envs/rpxdock/bin/python

import logging, os, rpxdock as rp, numpy as np

def get_opts():
   parser = rp.options.default_cli_parser()
   addkw = rp.options.add_argument_unless_exists(parser)
   addarg("--bodies", type=str, nargs='+', required=True)
   addarg("--other_bodies", type=str, nargs='+', default=[])
   addarg("--dump_scored_pdb", action='store_true', default=False)
   kw = parser.parse_args()
   return rp.options.process_cli_args(arg)

def score_onebody(hscore, **kw):
   kw = rp.Bunch(kw)
   for fn in kw.bodies:
      body = rp.Body(fn)
      iscores = hscore.score_matrix_intra(body, kw.wts)
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
      if kw.dump_scored_pdb:
         resscore = np.sum(iscores, axis=0)
         bfac_file = os.path.basename(fn) + '_rpx.pdb'
         body.dump_pdb(bfac_file, bfactor=resscore, use_orig_coords=True)
      print(f'{fn} sum {sumsum:7.2f} mean {meanmean:7.3f} sum(max) {summax:8.3f} ' +
            f'sum(mean) {summean:7.3f} {bfac_file}')

def score_twobody(hscore, **kw):
   kw = rp.Bunch(kw)
   for fn1 in kw.bodies:
      body1 = rp.Body(fn1)
      for fn2 in kw.other_bodies:
         body2 = Body(fn2)
         print(fn1, fn2, hscore.score(body1, body2, kw.wts))

def main():
   kw = get_opts()
   logging.info(f'weights: {kw.wts}')

   hscore = rp.RpxHier(kw.hscore_files, **kw)

   if kw.other_bodies:
      score_twobody(hscore, **kw)
   else:
      score_onebody(hscore, **kw)

if __name__ == '__main__':
   main()