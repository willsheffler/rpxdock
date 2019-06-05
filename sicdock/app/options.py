import argparse, functools
import numpy as np
from sicdock.util import cpu_count, Bunch

_iface_summary_methods = dict(min=np.min, sum=np.sum)

def add_argument_unless_exists(parser, *args, **kw):
   try:
      parser.add_argument(*args, **kw)
   except argparse.ArgumentError:
      pass

def default_cli_parser(parser=None):
   if parser is None:
      parser = argparse.ArgumentParser()
   addarg = functools.partial(add_argument_unless_exists, parser)
   addarg("--inputs", nargs="*", type=str, default=[])
   addarg("--ncpu", type=int, default=cpu_count())
   addarg("--nthread", type=int, default=0)
   addarg("--nprocess", type=int, default=0)
   addarg("--trial_run", action="store_true", default=False)
   addarg("--hscore_files", nargs="+", default=[])
   addarg("--max_trim", type=int, default=100)
   addarg("--nout", type=int, default=0)
   addarg("--nresl", type=int, default=5)
   addarg("--clashdis", type=float, default=3.5)
   addarg("--beam_size", type=int, default=1e5)
   addarg("--rmscut", type=float, default=3.0)
   addarg("--max_longaxis_dot_z", type=float, default=1.000001)
   addarg("--iface_summary", default="min")
   addarg("--weight_rpx", default=1.0)
   addarg("--weight_ncontact", default=0.01)
   addarg("--weight_plug", default=1.0)
   addarg("--weight_hole", default=1.0)
   H = "output file prefix. will output pickles for a base ResPairScore plus --hierarchy_depth hier XMaps"
   addarg("--output_prefix", nargs="?", default="auto", type=str, help=H)
   addarg("--dont_store_body_in_results", action="store_true", default=False)
   parser.has_sicdock_args = True
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
