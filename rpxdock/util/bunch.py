__all__ = ("Bunch", "bunchify", "unbunchify")

class Bunch(dict):
   def __init__(self, __arg_or_ns=None, **kw):
      if __arg_or_ns is not None:
         try:
            super().__init__(__arg_or_ns)
         except TypeError:
            super().__init__(vars(__arg_or_ns))
      self.update(kw)

   def __contains__(self, k):
      try:
         return dict.__contains__(self, k) or k in self.__dict__
      except:
         return False

   def __getattr__(self, k):
      try:
         # Throws exception if not in prototype chain
         return object.__getattribute__(self, k)
      except AttributeError:
         try:
            return self[k]
         except KeyError:
            return None

   def __setattr__(self, k, v):
      try:
         # Throws exception if not in prototype chain
         object.__getattribute__(self, k)
      except AttributeError:
         try:
            self[k] = v
         except:
            raise AttributeError(k)
      else:
         object.__setattr__(self, k, v)

   def __delattr__(self, k):
      try:
         # Throws exception if not in prototype chain
         object.__getattribute__(self, k)
      except AttributeError:
         try:
            del self[k]
         except KeyError:
            raise AttributeError(k)
      else:
         object.__delattr__(self, k)

   def copy(self):
      return Bunch.from_dict(super().copy())

   def toDict(self):
      return unbunchify(self)

   def sub(self, __BUNCH_SUB_ITEMS=None, **kw):
      if len(kw) is 0:
         if isinstance(__BUNCH_SUB_ITEMS, dict):
            kw = __BUNCH_SUB_ITEMS
         else:
            kw = vars(__BUNCH_SUB_ITEMS)
      newbunch = self.copy()
      for k, v in kw.items():
         if v is None and k in newbunch:
            del newbunch[k]
         else:
            newbunch.__setattr__(k, v)
      return newbunch

   def visit_remove_if(self, func, recurse=True, depth=0):
      toremove = list()
      for k, v in self.__dict__:
         if func(k, v, depth):
            toremove.append(k)
         elif isinstance(v, Bunch) and recurse:
            v.visit_remove_if(func, recurse, depth=depth + 1)
      for k, v in self.items():
         if func(k, v, depth):
            toremove.append(k)
         elif isinstance(v, Bunch) and recurse:
            v.visit_remove_if(func, recurse, depth=depth + 1)
      for k in toremove:
         self.__delattr__(k)

   def __add__(self, addme):
      newbunch = self.copy()
      for k, v in addme.items():
         if k in self:
            newbunch.__setattr__(k, self[k] + v)
         else:
            newbunch.__setattr__(k, v)
      return newbunch

   def __getstate__(self):
      return self.__dict__

   def __setstate__(self, d):
      self.__dict__.update(d)

   def __repr__(self):
      args = ", ".join(["%s=%r" % (key, self[key]) for key in self.keys()])
      return "%s(%s)" % (self.__class__.__name__, args)

   @staticmethod
   def from_dict(d):
      return bunchify(d)

def bunchify(x):
   if isinstance(x, dict):
      return Bunch(**x)
   elif isinstance(x, (list, tuple)):
      return type(x)(bunchify(v) for v in x)
   else:
      return x

def unbunchify(x):
   if isinstance(x, dict):
      return dict((k, unbunchify(v)) for k, v in x.items())
   elif isinstance(x, (list, tuple)):
      return type(x)(unbunchify(v) for v in x)
   else:
      return x
