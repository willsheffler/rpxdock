import sys, pytest
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

def print_input_args(args):
   print('==========================================')
   print('inputs            ', args.inputs)
   print('allowed_residues  ', args.allowed_residues)
   print('inputs1           ', args.inputs1)
   print('allowed_residues1 ', args.allowed_residues1)
   print('inputs2           ', args.inputs2)
   print('allowed_residues2 ', args.allowed_residues2)
   print('inputs3           ', args.inputs3)
   print('allowed_residues3 ', args.allowed_residues3)
   print('==========================================')

def test_inputs():

   args = defaults(process_args=False).sub(
      inputs=['foo', 'bar'],
      allowed_residues=['xfoo'],
   )
   args = process_cli_args(args)
   assert len(args.inputs) == len(args.allowed_residues)

   args = defaults(process_args=False).sub(
      inputs=['foo', 'bar'],
      allowed_residues=['xfoo'],
   )
   args = process_cli_args(args)
   assert len(args.inputs) == len(args.allowed_residues)

   args = defaults(process_args=False).sub(inputs=['foo', 'bar'])
   args = process_cli_args(args)
   print(args.allowed_residues)
   assert len(args.inputs) == len(args.allowed_residues)

   args = defaults(process_args=False).sub(
      inputs=['foo', 'bar'],
      allowed_residues=['xfoo'],
   )
   args = process_cli_args(args)
   assert len(args.inputs) == len(args.allowed_residues)

   args = defaults(process_args=False).sub(
      inputs=['foo', 'bar'],
      allowed_residues=[],
   )
   args = process_cli_args(args)
   assert args.allowed_residues == [None, None]

   args = defaults(process_args=False).sub(
      inputs=['foo', 'bar'],
      inputs1=['foo', 'bar'],
   )
   with pytest.raises(AssertionError):
      args = process_cli_args(args)

   args = defaults(process_args=False).sub(
      inputs=['foo', 'bar'],
      inputs1=['foo', 'bar'],
   )
   with pytest.raises(AssertionError):
      args = process_cli_args(args)

   args = defaults(process_args=False).sub(
      inputs1=['foo', 'bar'],
      allowed_residues1=['xfoo', 'xbar'],
      inputs2=['foo', 'bar'],
      allowed_residues2=['xfoo2'],
      inputs3=['foo', 'bar'],
      allowed_residues3=['xfoo', 'xbar'],
   )
   args = process_cli_args(args)
   assert len(args.inputs) == 3

   for i, a in zip(args.inputs, args.allowed_residues):
      assert len(i) == len(a)

   args = defaults(process_args=False).sub(
      inputs1=['foo', 'bar'],
      allowed_residues1=['xfoo1'],
      inputs2=['foo', 'bar'],
      # allowed_residues2=['xfoo2'],
      inputs3=['foo', 'bar'],
      allowed_residues3=['xfoo', 'xbar'],
   )
   args = process_cli_args(args)
   assert len(args.inputs) == 3
   assert len(args.allowed_residues) == 3
   assert args.allowed_residues[1] == [None, None]
   for i, a in zip(args.inputs, args.allowed_residues):
      assert len(i) == len(a)

if __name__ == '__main__':
   # test_defaults()
   # test_get_cli_args()
   test_inputs()