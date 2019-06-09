import sys, argparse, functools, logging
import numpy as np
from sicdock.util import cpu_count, Bunch

_iface_summary_methods = dict(min=np.min, sum=np.sum)

def add_argument_unless_exists(parser, *arg, **kw):
   try:
      parser.add_argument(*arg, **kw)
   except argparse.ArgumentError:
      pass

def default_cli_parser(parent=None):
   parser = parent if parent else argparse.ArgumentParser()
   addarg = functools.partial(add_argument_unless_exists, parser)
   addarg("--inputs", nargs="*", type=str, default=[])
   addarg("--ncpu", type=int, default=cpu_count())
   addarg("--nthread", type=int, default=0)
   addarg("--nprocess", type=int, default=0)
   addarg("--trial_run", action="store_true", default=False)
   addarg("--hscore_files", nargs="+", default=[])
   addarg("--max_trim", type=int, default=100)
   addarg("--trim_direction", type=str, default="NC")
   addarg("--nout_debug", type=int, default=0)
   addarg("--nout_top", type=int, default=10)
   addarg("--nout_each", type=int, default=1)
   addarg("--dump_pdbs", action='store_true', default=False)
   addarg("--suppress_dump_results", action='store_true', default=False)
   addarg("--nresl", type=int, default=5)
   addarg("--clashdis", type=float, default=3.5)
   addarg("--beam_size", type=int, default=1e5)
   addarg("--max_bb_redundancy", type=float, default=3.0)
   addarg("--max_longaxis_dot_z", type=float, default=1.000001)
   addarg("--iface_summary", default="min")
   addarg("--weight_rpx", type=float, default=1.0)
   addarg("--weight_ncontact", type=float, default=0.01)
   addarg("--weight_plug", type=float, default=1.0)
   addarg("--weight_hole", type=float, default=1.0)
   addarg("--grid_resolution_cart_angstroms", type=float, default=1)
   addarg("--grid_resolution_ori_degrees", type=float, default=1)
   H = "output file prefix. will output pickles for a base ResPairScore plus --hierarchy_depth hier XMaps"
   addarg("--output_prefix", nargs="?", default="auto", type=str, help=H)
   addarg("--dont_store_body_in_results", action="store_true", default=False)
   addarg("--loglevel", default='INFO')
   parser.has_sicdock_args = True
   return parser

def get_cli_args(argv=None):
   parser = default_cli_parser()
   argv = sys.argv[1:] if argv is None else argv
   argv = make_argv_with_atfiles(argv)
   arg = parser.parse_args(argv)
   return process_cli_args(arg)

def defaults():
   return get_cli_args([])

def process_cli_args(arg):
   arg = Bunch(arg)

   arg.iface_summary = _iface_summary_methods[arg.iface_summary]

   _extract_weights(arg)

   numeric_level = getattr(logging, arg.loglevel.upper(), None)
   if not isinstance(numeric_level, int):
      raise ValueError('Invalid log level: %s' % arg.loglevel)
   logging.getLogger().setLevel(level=numeric_level)

   return arg

def make_argv_with_atfiles(argv=None):
   if argv is None: argv = sys.argv[1:]
   for a in argv.copy():
      if not a.startswith('@'): continue
      argv.remove(a)
      with open(a[1:]) as inp:
         newargs = []
         for l in inp:
            # last char in l is newline, so [:-1] ok
            newargs.extend(l[:l.find("#")].split())
         argv = newargs + argv
   return argv

def _extract_weights(arg):
   pref = 'weight_'
   wts = Bunch()
   todel = list()
   for k in arg:
      if k.startswith(pref):
         wtype = k.replace(pref, '')
         wts[wtype] = arg[k]
         todel.append(k)
   for k in todel:
      del arg[k]
   arg.wts = wts
