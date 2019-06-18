import os, sys, time, _pickle, types, logging, copy, rpxdock

log = logging.getLogger(__name__)

class Cache(dict):
   def __init__(self):
      self.checkpoint()
      self._nodump = set()

   def checkpoint(self):
      self._checkpoint = set(self.keys())

   def keys_have_changed_since_checkpoint(self):
      return set(self.keys()) != self._checkpoint

   def __call__(self, fun, *args, _force_reload=False, _saved_only=False, _nodump=False,
                _key=None, **kw):
      if _key is None:
         _key = fun.__name__, repr(args), repr(kw)
      try:
         assert not _force_reload
         val = self[_key]
      except (KeyError, AssertionError):
         if _saved_only:
            raise ValueError(f'no cache entry for {_key}')
         log.info(f"Cache miss, computing {_key}")
         val = fun(*args, **kw)
         self[_key] = val
      if _nodump:
         self._nodump.add(_key)
      return val

   def remove(self, fun, *args, _force_reload=False, **kw):
      _key = fun.__name__, repr(args), repr(kw)
      del self[_key]

   def save(self, fname, force=False):
      fexists = os.path.exists(fname)
      changed = self.keys_have_changed_since_checkpoint()
      if fname and (force or not fexists or changed):
         dname = os.path.dirname(fname)
         os.makedirs(dname if dname else '.', exist_ok=True)
         tosave = copy.copy(self)
         for no in self._nodump:
            del tosave[no]
         with open(fname, 'wb') as out:
            _pickle.dump(tosave, out)
         return True
      return False

   def load(self, fname, strict=True):
      try:
         with open(fname, 'rb') as inp:
            other = _pickle.load(inp)
         self.clear()
         self.update(other)
         self.checkpoint()
      except (FileNotFoundError, TypeError, EOFError) as e:
         if strict:
            raise e

GLOBALCACHE = Cache()
if sys.argv and sys.argv[0] in ['ipython', 'test_server']:
   if not hasattr(os, "__HACK_MULTIRUN_CACHE"):
      os._HACK_MULTIRUN_CACHE = Cache()
   GLOBALCACHE = os._HACK_MULTIRUN_CACHE

def NOCACHE(fun, *args, **kw):
   return run(*args, **kw)
