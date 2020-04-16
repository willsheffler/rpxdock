from abc import ABC

class BodyPairEvaluator:
   def __init__(self, components):
      self.components = components
      self.filters = sorted([c for c in components if c.is_filter], key=lambda x: x.priority)
      self.trimmers = sorted([c for c in components if c.is_trimmer], key=lambda x: x.priority)
      self.scorefuncs = sorted([c for c in components if c.is_score], key=lambda x: x.priority)

   # def isok(self):
   #    return True

   def __call__(self, bodies, positions, **kw):
      pass

   def __str__(self):
      return f'BodyPairEvaluator components: {self.components}'

class BodyPairEvalComponent(ABC):
   score_types = []
   extra_types = []

   def __init__(self, name=None, priority: float = 0):
      super().__init__()
      self.priority = priority
      self.is_filter = hasattr(self, 'filter')
      self.is_trimmer = hasattr(self, 'trim')
      self.is_score = hasattr(self, 'score')
      self.name = name if name else self.__class__.__name__
      self.check_validity()

   def check_validity(self):
      if not (self.is_score or self.is_filter or self.is_trimmer):
         raise TypeError(
            f'BodyPairEvalComponent {self.name} must define one of filter, trim or score')
      if self.is_score != bool(self.score_types):
         raise TypeError(
            f'BodyPairEvalComponent {self.name} with score() must specify score_type(s)')
      if not isinstance(self.score_types, (tuple, list)):
         raise TypeError(f'{self.name}.score_types must be list or tuple')
      if not all(isinstance(s, str) for s in self.score_types):
         raise TypeError(f'{self.name}.score_types contents must be str')
      if not all(isinstance(s, str) for s in self.extra_types):
         raise TypeError(f'{self.name}.extra_types contents must be str')

   def __repr__(self):
      return f'BodyPairEvalComponent {self.name}'
