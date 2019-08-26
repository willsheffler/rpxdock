class ScoreFunction:
   def __init__(self, components: list):
      self.components = components
      self.filters = sorted([c for c in components if c.is_filter], key=lambda x: x.priority)
      self.trimmers = sorted([c for c in components if c.is_trimmer], key=lambda x: x.priority)
      self.scorefuncs = sorted([c for c in components if c.is_score], key=lambda x: x.priority)

   def isok(self):
      return True

   def __call__(self, bodies, positions, **kw):
      pass

   def __str__(self):
      return f'ScoreFunction components: {self.components}'
