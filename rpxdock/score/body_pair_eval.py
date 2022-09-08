import abc, itertools as it

class BodyPairEvaluator:
   def __init__(self, components):
      self.components = components
      self.filters = sorted([c for c in components if c.is_filter], key=lambda x: x.priority)
      self.trimmers = sorted([c for c in components if c.is_trimer], key=lambda x: x.priority)
      self.scorers = sorted([c for c in components if c.is_scorer], key=lambda x: x.priority)
      self.check_and_populate_labels()

   def filter(self, bodies, positions, **kw):
      data = dict()
      passed = np.ones(len(positions), dtype=np.bool)
      for f in filters:
         passthis, d = f(bodies, positions[passed], body_pair_evaluator=self, **kw)
         passed[passed] = passthis
         data.update(d)
      return passed, data

   def trim(self, bodies, positions, **kw):
      data = dict()
      bounds = np.zeros([len(positions), len(bodies), 2], dtype=np.bool)
      for i, b in enumerate(bodies):
         bounds[:, i, 1] = len(b)
      for f in filters:

         todo

         passthis, d = f(bodies, positions[passed], body_pair_evaluator=self, **kw)
         passed[passed] = passthis
         data.update(d)
      return passed, data

   def score(self, bodies, positions, **kw):
      return {k: v for k, v in f(self, bodies, positions, **kw) for f in self.scorers}

   def __call__(self, bodies, positions, **kw):
      pass

   def __str__(self):
      return f'BodyPairEvaluator components: {self.components}'

   def check_and_populate_labels(self):
      self.data_labels = []
      tmp = sorted(sum([c.data_labels for c in self.components], []))
      if tmp:
         counts = {key: len(list(group)) for key, group in it.groupby(tmp)}
         err = []
         for k, c in counts.items():
            if c != 1:
               err.append(f'metric label {k} defined {c} times')
         if err:
            raise TypeError('metric label must be defined only once: ' + ', '.join(err))
         self.data_labels = list(counts.keys())

class BodyPairEvalComponent(abc.ABC):
   def __init__(self, data_labels, name=None, priority: float = 0):
      super().__init__()
      if not data_labels:
         raise ValueError(f'BodyPairEvalComponent must specify data_labels')
      self.data_labels = data_labels
      self.priority = priority
      self.is_filter = hasattr(self, 'filter')
      self.is_trimer = hasattr(self, 'trim')
      self.is_scorer = hasattr(self, 'score')
      self.name = name if name else self.__class__.__name__
      self.check_validity()

   def check_validity(self):
      if not (self.is_scorer or self.is_filter or self.is_trimer):
         raise TypeError(
            f'BodyPairEvalComponent {self.name} must define one of filter, trim or score')
      if not self.data_labels:
         raise TypeError(f'BodyPairEvalComponent "{self.name}" must specify data_labels')
      if not isinstance(self.data_labels, (tuple, list)):
         raise TypeError(f'{self.name}.data_labels must be list or tuple')
      if not all(isinstance(s, str) for s in self.data_labels):
         raise TypeError(f'{self.name}.data_labels contents must be str')
      if not all(isinstance(s, str) for s in self.data_labels):
         raise TypeError(f'{self.name}.extra_fields contents must be str')

   def __repr__(self):
      return f'BodyPairEvalComponent {self.name}'
