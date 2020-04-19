import abc, itertools as it

class BodyPairEvaluator:
   def __init__(self, components):
      self.components = components
      self.filters = sorted([c for c in components if c.is_filter], key=lambda x: x.priority)
      self.trimmers = sorted([c for c in components if c.is_trimer], key=lambda x: x.priority)
      self.scorefuncs = sorted([c for c in components if c.is_score], key=lambda x: x.priority)

      self.score_fields = []
      tmp = sorted(sum([c.score_fields for c in components], []))
      if tmp:
         counts = {key: len(list(group)) for key, group in it.groupby(tmp)}
         err = []
         for k, c in counts.items():
            if c != 1:
               err.append(f'score_field {k} defined {c} times')
         if err:
            raise TypeError('score_field must be defined only once: ' + ', '.join(err))
         self.score_fields = list(counts.keys())

      self.extra_fields = []
      tmp = sorted(sum([c.extra_fields for c in components], []))
      if tmp:
         counts = {key: len(list(group)) for key, group in it.groupby(tmp)}
         err = []
         for k, c in counts.items():
            if c != 1:
               err.append(f'score_field {k} defined {c} times')
         if err:
            raise TypeError('score_field must be defined only once: ' + ', '.join(err))
         self.extra_fields = list(counts.keys())

   # def isok(self):
   #    return True

   def __call__(self, bodies, positions, **kw):
      pass

   def __str__(self):
      return f'BodyPairEvaluator components: {self.components}'

class BodyPairEvalComponent(abc.ABC):
   score_fields = []
   extra_fields = []

   def __init__(self, name=None, priority: float = 0):
      super().__init__()
      self.priority = priority
      self.is_filter = hasattr(self, 'filter')
      self.is_trimer = hasattr(self, 'trim')
      self.is_score = hasattr(self, 'score')
      self.name = name if name else self.__class__.__name__
      self.check_validity()

   def check_validity(self):
      if not (self.is_score or self.is_filter or self.is_trimer):
         raise TypeError(
            f'BodyPairEvalComponent {self.name} must define one of filter, trim or score')
      if self.is_score != bool(self.score_fields):
         raise TypeError(
            f'BodyPairEvalComponent {self.name} with score() must specify score_fields')
      if not isinstance(self.score_fields, (tuple, list)):
         raise TypeError(f'{self.name}.score_fields must be list or tuple')
      if not all(isinstance(s, str) for s in self.score_fields):
         raise TypeError(f'{self.name}.score_fields contents must be str')
      if not all(isinstance(s, str) for s in self.extra_fields):
         raise TypeError(f'{self.name}.extra_fields contents must be str')

   def __repr__(self):
      return f'BodyPairEvalComponent {self.name}'
