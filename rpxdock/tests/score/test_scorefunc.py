import pytest
from rpxdock.score import *

class BadComp(ScoreComponent):
   def __init__(self):
      super().__init__()

class BadScoreComp(ScoreComponent):
   def __init__(self):
      super().__init__()

   def score(self):
      pass

class BadScoreComp2(ScoreComponent):
   score_types = ['foo']

   def __init__(self):
      super().__init__()

   def trim(self):
      pass

class BadScoreComp2(ScoreComponent):
   score_types = ['foo', 'bar']

   def __init__(self):
      super().__init__()

   def trim(self):
      pass

class BadScoreComp3(ScoreComponent):
   score_types = ['foo']

   def __init__(self):
      super().__init__()

   def trim(self):
      pass

class BadScoreComp4(ScoreComponent):
   score_types = [1]

   def __init__(self):
      super().__init__()

   def score(self):
      pass

class FilterComp(ScoreComponent):
   def __init__(self, name=None, priority=0):
      super().__init__(name=name, priority=priority)

   def filter(self):
      pass

class TrimComp(ScoreComponent):
   def __init__(self):
      super().__init__()

   def trim(self):
      pass

class ScoreComp(ScoreComponent):
   score_types = ['foo']

   def __init__(self):
      super().__init__()

   def score(self):
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
   func = ScoreFunction([f1, f2, f3, f4, t, t, s])
   assert len(func.filters) is 4
   assert len(func.trimmers) is 2
   assert len(func.scorefuncs) is 1
   assert func.filters == [f4, f2, f3, f1]

def main():
   test_score_component_base()
   test_score_function()

if __name__ == '__main__':
   main()
