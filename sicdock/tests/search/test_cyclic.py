from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
import _pickle, threading, os, argparse, sys
from time import perf_counter
import numpy as np

import sicdock
from sicdock.motif import HierScore
from sicdock.util import GLOBALCACHE as HC
from sicdock.data import datadir
from sicdock.util import load, Bunch, load_threads
from sicdock.io.io_body import dump_pdb_from_bodies
from sicdock.geom import symframes
from sicdock.search import make_cyclic, concat_results

def test_make_cyclic(hscore, cli_args=dict()):
   args = sicdock.app.defaults()
   args.nresl = 5
   args.wts = Bunch(ncontact=0.1, rpx=0.0)
   args.beam_size = 1e3
   args.max_bb_redundancy = 3.0
   # args.max_longaxis_dot_z = 0.5
   args.max_trim = 0
   # args.executor = ThreadPoolExecutor(args.nthread if args.nthread else args.ncpu)
   args.sub(cli_args)

   body = sicdock.body.Body(datadir + "/pdb/DHR14.pdb")
   result = make_cyclic(body, "C3", hscore, **args)
   result = concat_results([result])
   print(result.attrs)

def main():
   hscore = HierScore(load(small_hscore_fnames))
   test_make_cyclic(hscore)

if __name__ == "__main__":
   main()
