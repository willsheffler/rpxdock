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

class BadTrimComp(BodyPairEvalComponent):
   def __init__(self):
      super().__init__()

   def trim(self):
      pass

class BadFilterComp(BodyPairEvalComponent):
   def __init__(self):
      super().__init__()

   def filter(self):
      pass

class BadComp2(BodyPairEvalComponent):
   def __init__(self):
      super().__init__()

   @property
   def data_labels(self):
      return ['foo']

class BadScoreComp2(BodyPairEvalComponent):
   def __init__(self):
      super().__init__()

   def score(self):
      pass

   @property
   def data_labels(self):
      data_labels = [1]

class FilterComp(BodyPairEvalComponent):
   def __init__(self, name=None, priority=0, data_labels=None):
      super().__init__(name=name, priority=priority, data_labels=data_labels)

   def filter(self):
      pass

class TrimComp(BodyPairEvalComponent):
   def __init__(self, data_labels):
      super().__init__(data_labels=data_labels)

   def trim(self):
      pass

class ScoreComp(BodyPairEvalComponent):
   def __init__(self):
      super().__init__(data_labels=['score1', 'score2'])

   def score(self):
      pass

class AllComp(BodyPairEvalComponent):
   def __init__(self):
      super().__init__(data_labels='all')

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
      BadComp2()
   with pytest.raises(TypeError):
      BadFilterComp()
   with pytest.raises(TypeError):
      BadTrimComp()
   with pytest.raises(TypeError):
      BadScoreComp()
   with pytest.raises(TypeError):
      BadScoreComp2()
   f = FilterComp(data_labels=['foo'])
   t = TrimComp(['bar'])
   s = ScoreComp()

def test_score_function():
   f1 = FilterComp(name='fc9', priority=9, data_labels=['is_stupid'])
   f2 = FilterComp(name='fc3', priority=3, data_labels=['is_dump'])
   f3 = FilterComp(name='fc5', priority=5, data_labels=['is_good'])
   f4 = FilterComp(name='fc1', priority=1, data_labels=['is_bad'])
   t1 = TrimComp(data_labels=['lb1', 'ub1'])
   t2 = TrimComp(data_labels=['lb2', 'ub2'])
   s = ScoreComp()  # already knows it's data labels
   func = BodyPairEvaluator([f1, f2, f3, f4, t2, t1, s])
   assert len(func.filters) is 4
   assert len(func.trimmers) is 2
   assert len(func.scorers) is 1
   assert func.filters == [f4, f2, f3, f1]

   with pytest.raises(TypeError):
      BodyPairEvaluator([ScoreComp(), ScoreComp()])

def main():
   test_score_component_base()
   test_score_function()

if __name__ == '__main__':
   main()
