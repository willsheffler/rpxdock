import sys
from rpxdock.app.options import *

def test_defaults():
   args = defaults()
   print(args)
   assert args.wts.rpx == 1.0
   assert args.wts.hole == 1.0
   assert args.inputs == []

def test_get_cli_args():
   sys.argv = "prog --inputs foo bar baz".split()
   args = get_cli_args()
   assert args.inputs == 'foo bar baz'.split()

if __name__ == '__main__':
   test_defaults()
   test_get_cli_args()
