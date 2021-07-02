import pytest
from rpxdock.score import *

class BadComp(BodyPairEvalComponent):
   def __init__(self):
      super().__init__()

class BadScoreComp(BodyPairEvalComponent):
   def __init__(self):
      super().__init__()

   def score(self):
      pass

class BadScoreComp2(BodyPairEvalComponent):
   score_fields = ['foo']

   def __init__(self):
      super().__init__()

   def trim(self):
      pass

class BadScoreComp5(BodyPairEvalComponent):
   score_fields = ['foo', 'bar']

   def __init__(self):
      super().__init__()

   def trim(self):
      pass

class BadScoreComp3(BodyPairEvalComponent):
   score_fields = ['foo']

   def __init__(self):
      super().__init__()

   def trim(self):
      pass

class BadScoreComp4(BodyPairEvalComponent):
   score_fields = [1]

   def __init__(self):
      super().__init__()

   def score(self):
      pass

class FilterComp(BodyPairEvalComponent):
   def __init__(self, name=None, priority=0):
      super().__init__(name=name, priority=priority)

   def filter(self):
      pass

class TrimComp(BodyPairEvalComponent):
   def __init__(self):
      super().__init__()

   def trim(self):
      pass

class ScoreComp(BodyPairEvalComponent):
   score_fields = ['foo']

   def __init__(self):
      super().__init__()

   def score(self):
      pass

class AllComp(BodyPairEvalComponent):
   score_fields = ['foo']

   def __init__(self):
      super().__init__()

   def score(self):
      pass

   def trim(self):
      pass

   def filter(self):
      pass

def test_score_component_base():
   with pytest.raises(TypeError):
      BadComp()
   with pytest.raises(TypeError):
      BadScoreComp()
   with pytest.raises(TypeError):
      BadScoreComp2()
   with pytest.raises(TypeError):
      BadScoreComp3()
   with pytest.raises(TypeError):
      BadScoreComp4()
   with pytest.raises(TypeError):
      BadScoreComp5()
   f = FilterComp()
   t = TrimComp()
   s = ScoreComp()

def test_score_function():
   f1 = FilterComp('fc9', 9)
   f2 = FilterComp('fc3', 3)
   f3 = FilterComp('fc5', 5)
   f4 = FilterComp('fc1', 1)
   t = TrimComp()
   s = ScoreComp()
   func = BodyPairEvaluator([f1, f2, f3, f4, t, t, s])
   assert len(func.filters) is 4
   assert len(func.trimmers) is 2
   assert len(func.scorefuncs) is 1
   assert func.filters == [f4, f2, f3, f1]

   with pytest.raises(TypeError):
      BodyPairEvaluator([ScoreComp(), ScoreComp()])

def main():
   test_score_component_base()
   test_score_function()

if __name__ == '__main__':
   main()
