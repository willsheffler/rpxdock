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

def test_parse_termini():
   kw = defaults(process_args=False).sub(
      inputs1=['foo', 'bar'],
      inputs2=['foo'],
      inputs3=['foo', 'bar'],
      term_access1= [True, False, False, True],
      term_access2 = [True, False],
      termini_dir1 = [True],
      termini_dir3 = [False, False, True, True]
   )
   kw = process_cli_args(kw)
   assert len(kw.inputs) == 3
   assert len(kw.inputs) == len(kw.termini_dir)
   assert len(kw.inputs) == len(kw.term_access)
   assert kw.term_access[0] == kw.term_access1 == [[True, False], [False, True]]
   assert kw.term_access[1] == kw.term_access2 == [[True, False]]   
   assert kw.term_access[2] == kw.term_access3 == [[False, False], [False, False]]   

   assert kw.termini_dir[0] == kw.termini_dir1 == [[True, True], [True, True]]
   assert kw.termini_dir[1] == kw.termini_dir2 == [[None, None]]
   assert kw.termini_dir[2] == kw.termini_dir3 == [[False, False], [True, True]]

   kw = defaults(process_args=False).sub(
      inputs1=['foo', 'bar'],
      inputs2=['foo'],
      term_access1= [True, False],
      term_access2 = [True]
   )
   # Should have issue with term_access1
   with pytest.raises(AssertionError):
      kw = process_cli_args(kw)

   kw = defaults(process_args=False).sub(
      inputs1=['foo'],
      inputs2=['foo', 'bar'],
      termini_dir1 = [False, True],
      termini_dir2 = [False, True]
   )
   # Should have issue with termini_dir2
   with pytest.raises(AssertionError):
      kw = process_cli_args(kw)

def test_str2bool():
   assert str2bool('True')
   with pytest.raises(argparse.ArgumentTypeError):
      assert str2bool('8')
   assert str2bool('1')
   with pytest.raises(argparse.ArgumentTypeError):
      assert str2bool('-1')
   assert not str2bool('faLse')
   assert not str2bool('0')

   assert str2bool(['t', 'f', 'no', 'yes']) == [True, False, False, True]

def test_dir_plus_bool():
   with pytest.raises(argparse.ArgumentTypeError):
      assert dir_plus_bool('inside')
      assert dir_plus_bool('-1')
   assert dir_plus_bool("In")   
   assert dir_plus_bool("DOWN")
   assert not dir_plus_bool("Out")
   assert not dir_plus_bool("NO") 

   assert None == dir_plus_bool("NA")
   assert None == dir_plus_bool("NoNe")
   with pytest.raises(argparse.ArgumentTypeError):
      assert dir_plus_bool("n/a")

   assert (dir_plus_bool(['t', 'f', 'no', 'yes', '1', '0']) == 
                  [True, False, False, True, True, False])

def test_inputs_read_allowed_res():
   kw = defaults(process_args=False).sub(
      inputs=['foo'],
      allowed_residues=[rp.data.datadir + '/example_allowed_residues.txt'],
   )
   kw = process_cli_args(kw)
   assert len(kw.allowed_residues) == 1
   assert kw.allowed_residues[0](range(1, 11)) == {3, 6, 7, 8, 9, 10}
   assert kw.allowed_residues[0](range(1, 21)) == {3, 6, 7, 8, 9, 10, 11, 17, 18, 19, 20}
   kw = defaults(process_args=False).sub(inputs=['foo'])
   kw = process_cli_args(kw)
   assert kw.allowed_residues[0] is None

if __name__ == '__main__':
   # test_defaults()
   # test_get_cli_args()
   # test_inputs_read_allowed_res()
   # test_str2bool()
   test_dir_plus_bool()
   test_parse_termini()
   # test_inputs()
