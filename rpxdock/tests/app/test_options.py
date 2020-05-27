import sys, pytest, rpxdock as rp
from rpxdock.app.options import *

def test_defaults():
   kw = defaults()
   print(kw)
   assert kw.wts.rpx == 1.0
   assert kw.wts.hole == 1.0
   assert kw.inputs == []

def test_get_cli_args():
   sys.argv = "prog --inputs foo bar baz".split()
   kw = get_cli_args()
   assert kw.inputs == 'foo bar baz'.split()

def print_input_args(kw):
   print('==========================================')
   print('inputs            ', kw.inputs)
   print('allowed_residues  ', kw.allowed_residues)
   print('inputs1           ', kw.inputs1)
   print('allowed_residues1 ', kw.allowed_residues1)
   print('inputs2           ', kw.inputs2)
   print('allowed_residues2 ', kw.allowed_residues2)
   print('inputs3           ', kw.inputs3)
   print('allowed_residues3 ', kw.allowed_residues3)
   print('==========================================')

def test_inputs():

   kw = defaults(process_args=False).sub(
      inputs=['foo', 'bar'],
      allowed_residues=['xfoo'],
   )
   kw = process_cli_args(kw, read_allowed_res_files=False)
   assert len(kw.inputs) == len(kw.allowed_residues)

   kw = defaults(process_args=False).sub(
      inputs=['foo', 'bar'],
      allowed_residues=['xfoo'],
   )
   kw = process_cli_args(kw, read_allowed_res_files=False)
   assert len(kw.inputs) == len(kw.allowed_residues)

   kw = defaults(process_args=False).sub(inputs=['foo', 'bar'])
   kw = process_cli_args(kw, read_allowed_res_files=False)
   assert len(kw.inputs) == len(kw.allowed_residues)

   kw = defaults(process_args=False).sub(
      inputs=['foo', 'bar'],
      allowed_residues=['xfoo'],
   )
   kw = process_cli_args(kw, read_allowed_res_files=False)
   assert len(kw.inputs) == len(kw.allowed_residues)

   kw = defaults(process_args=False).sub(
      inputs=['foo', 'bar'],
      allowed_residues=[],
   )
   kw = process_cli_args(kw, read_allowed_res_files=False)
   assert kw.allowed_residues == [None, None]

   kw = defaults(process_args=False).sub(
      inputs=['foo', 'bar'],
      inputs1=['foo', 'bar'],
   )
   with pytest.raises(AssertionError):
      kw = process_cli_args(kw, read_allowed_res_files=False)

   kw = defaults(process_args=False).sub(
      inputs=['foo', 'bar'],
      inputs1=['foo', 'bar'],
   )
   with pytest.raises(AssertionError):
      kw = process_cli_args(kw, read_allowed_res_files=False)

   kw = defaults(process_args=False).sub(
      inputs1=['foo', 'bar'],
      allowed_residues1=['xfoo', 'xbar'],
      inputs2=['foo', 'bar'],
      allowed_residues2=['xfoo2'],
      inputs3=['foo', 'bar'],
      allowed_residues3=['xfoo', 'xbar'],
   )
   kw = process_cli_args(kw, read_allowed_res_files=False)
   assert len(kw.inputs) == 3

   for i, a in zip(kw.inputs, kw.allowed_residues):
      assert len(i) == len(a)

   kw = defaults(process_args=False).sub(
      inputs1=['foo', 'bar'],
      allowed_residues1=['xfoo1'],
      inputs2=['foo', 'bar'],
      # allowed_residues2=['xfoo2'],
      inputs3=['foo', 'bar'],
      allowed_residues3=['xfoo', 'xbar'],
   )
   kw = process_cli_args(kw, read_allowed_res_files=False)
   assert len(kw.inputs) == 3
   assert len(kw.allowed_residues) == 3
   assert kw.allowed_residues[1] == [None, None]
   for i, a in zip(kw.inputs, kw.allowed_residues):
      assert len(i) == len(a)

def test_inputs_read_allowed_res():
   kw = defaults(process_args=False).sub(
      inputs=['foo'],
      allowed_residues=[rp.data.datadir + '/example_allowed_residues.txt'],
   )
   kw = process_cli_args(kw)
   assert len(kw.allowed_residues) is 1
   assert kw.allowed_residues[0](range(1, 11)) == {3, 6, 7, 8, 9, 10}
   assert kw.allowed_residues[0](range(1, 21)) == {3, 6, 7, 8, 9, 10, 11, 17, 18, 19, 20}
   kw = defaults(process_args=False).sub(inputs=['foo'])
   kw = process_cli_args(kw)
   assert kw.allowed_residues[0] is None

if __name__ == '__main__':
   # test_defaults()
   # test_get_cli_args()
   test_inputs_read_allowed_res()
