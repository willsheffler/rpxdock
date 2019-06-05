import _pickle, os, multiprocessing, threading, copy, hashlib
from collections import abc
import numpy as np, xarray as xr

def load(f, verbose=True):
   if isinstance(f, str):
      if verbose: print('loading', f)
      with open(f, "rb") as inp:
         return _pickle.load(inp)
   return [load(x) for x in f]

def dump(obj, f):
   d = os.path.dirname(f)
   if d: os.makedirs(d, exist_ok=True)
   with open(f, "wb") as out:
      return _pickle.dump(obj, out)

def cpu_count():
   try:
      return int(os.environ["SLURM_CPUS_ON_NODE"])
   except:
      return multiprocessing.cpu_count()

def hash_str_to_int(s):
   if isinstance(s, str):
      s = s.encode()
   buf = hashlib.sha1(s).digest()[:8]
   return int(abs(np.frombuffer(buf, dtype="i8")[0]))

def sanitize_for_pickle(data):
   data = copy.copy(data)
   if isinstance(data, (np.ndarray, xr.Dataset, xr.DataArray, int, float, str)):
      pass
   elif isinstance(data, abc.MutableMapping):
      for k, v in data.items():
         data[k] = sanitize_for_pickle(v)
   elif isinstance(data, abc.MutableSequence):
      for i, v in enumerate(data):
         data[i] = sanitize_for_pickle(v)
   elif isinstance(data, tuple):
      data = tuple(sanitize_for_pickle(list(data)))
   elif isinstance(data, abc.Set):
      data = data.__class__(sanitize_for_pickle(list(data)))
   elif data is None:
      pass
   else:
      m = data.__module__ if hasattr(data, '__module__') else "unknown_module"
      if hasattr(data, '__name__'):
         n = data.__name__
      elif hasattr(data, '__class__'):
         n = f'{data.__class__.__name__}<instance>'
      else:
         n = 'unknown_name'
      if hasattr(n, '__str__'):
         n += '::' + str(data)
      data = m + '.' + n
   return data

class ThreadLoader(threading.Thread):
   def __init__(self, fname):
      super().__init__(None, None, None)
      self.fname = fname

   def run(self):
      self.result = load(self.fname, verbose=False)
      print("loaded", self.fname)

def load_threads(fnames):
   threads = [ThreadLoader(f) for f in fnames]
   [t.start() for t in threads]
   [t.join() for t in threads]
   return [t.result for t in threads]

class MultiThreadLoader(threading.Thread):
   def __init__(self, fnames):
      super().__init__(None, None, None)
      self.fnames = fnames

   def run(self):
      self.result = load_threads(self.fnames)
