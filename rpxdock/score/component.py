from abc import ABC, abstractmethod

class ScoreComponent(ABC):
   score_types = []
   extra_types = []

   def __init__(self, name=None, priority: float = 0):
      super().__init__()
      self.priority = priority
      self.is_filter = hasattr(self, 'filter')
      self.is_trimmer = hasattr(self, 'trim')
      self.is_score = hasattr(self, 'score')
      self.name = name if name else self.__class__.__name__
      if not (self.is_score or self.is_filter or self.is_trimmer):
         raise TypeError(f'ScoreComponent {self.name} must define one of filter, trim or score')
      if self.is_score != bool(self.score_types):
         raise TypeError(f'ScoreComponent {self.name} must define score and score_type together')
      if not isinstance(self.score_types, (tuple, list)):
         raise TypeError(f'{self.name}.score_types must be list or tuple')
      if not all(isinstance(s, str) for s in self.score_types):
         raise TypeError(f'{self.name}.score_types contents must be str')
      if not all(isinstance(s, str) for s in self.extra_types):
         raise TypeError(f'{self.name}.extra_types contents must be str')

   def __repr__(self):
      return f'ScoreComponent {self.name}'
