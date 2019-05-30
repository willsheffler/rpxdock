import argparse
import numpy as np
from sicdock.util import cpu_count, Bunch

_iface_summary_methods = dict(min=np.min, sum=np.sum)

def default_cli_parser(parser=None):
   if parser is None:
      parser = argparse.ArgumentParser()

   parser.add_argument("--inputs", nargs="*", type=str, default=[])
   parser.add_argument("--ncpu", type=int, default=cpu_count())
   parser.add_argument("--nthread", type=int, default=0)
   parser.add_argument("--nprocess", type=int, default=0)
   parser.add_argument("--trial_run", action="store_true", default=False)
   parser.add_argument("--hscore_files", nargs="+", default=[])
   parser.add_argument("--max_trim", type=int, default=100)
   parser.add_argument("--nout", type=int, default=10)
   parser.add_argument("--nresl", type=int, default=5)
   parser.add_argument("--clashdis", type=float, default=3.5)
   parser.add_argument("--beam_size", type=int, default=1e5)
   parser.add_argument("--rmscut", type=float, default=3.0)
   parser.add_argument("--max_longaxis_dot_z", type=float, default=1.000001)
   parser.add_argument("--iface_summary", default="min")
   parser.add_argument("--weight_rpx", default=1.0)
   parser.add_argument("--weight_ncontact", default=0.01)
   parser.add_argument("--weight_plug", default=1.0)
   parser.add_argument("--weight_hole", default=1.0)
   H = "output file prefix. will output pickles for a base ResPairScore plus --hierarchy_depth hier XMaps"
   parser.add_argument("--out_prefix", nargs="?", default="auto", type=str, help=H)

   return parser

def get_cli_args(argv=None):
   parser = default_cli_parser()
   if argv is None: args = parser.parse_args()
   else: args = parser.parse_args(argv)
   return process_cli_args(args)

def defaults():
   return get_cli_args([])

def process_cli_args(args):
   args = Bunch(args)
   args.iface_summary = _iface_summary_methods[args.iface_summary]
   _extract_weights(args)
   return args

def _extract_weights(args):
   pref = 'weight_'
   wts = Bunch()
   todel = list()
   for k in args:
      if k.startswith(pref):
         wtype = k.replace(pref, '')
         wts[wtype] = args[k]
         todel.append(k)
   for k in todel:
      del args[k]
   args.wts = wts
