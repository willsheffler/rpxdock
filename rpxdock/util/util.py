import _pickle, os, multiprocessing, threading, copy, hashlib, logging, concurrent, time, gzip, bz2, lzma, zipfile
from collections import abc
import numpy as np, xarray as xr

log = logging.getLogger(__name__)

def load(f, verbose=True):
   if isinstance(f, str):
      if verbose: log.debug(f'loading{f}')

      if f.endswith('.gz'):
         readfun = gzip.open
      elif f.endswith('.bz') or f.endswith('.bz2'):
         readfun = bz2.open
      elif f.endswith('.xz') or f.endswith('.lzma'):
         readfun = lzma.open
      elif f.endswith('.zip'):
         readfun = zipfile.Zipfile
      else:
         readfun = open
      with readfun(f, "rb") as inp:
         return _pickle.load(inp)
   return [load(x) for x in f]

def dump(thing, f):
   d = os.path.dirname(f)
   if d: os.makedirs(d, exist_ok=True)
   with open(f, "wb") as out:
      return _pickle.dump(thing, out)

def dump_str(string, f):
   d = os.path.dirname(f)
   if d: os.makedirs(d, exist_ok=True)
   if isinstance(string, (list, tuple)):
      string = '\n'.join(string)
   with open(f, "wb") as out:
      out.write(string.encode())
      out.write(b'\n')

def num_digits(n):
   isarray = isinstance(n, np.ndarray)
   if not isarray: n = np.array([n])
   absn = np.abs(n.astype('i8'))
   absn[absn == 0] = 1  # same num digits, avoid log problems
   ndig = 1 + np.floor(np.log10(absn)).astype('i8')
   ndig[absn == 0] = 1
   ndig[n < 0] += 1
   if not isarray and len(ndig) == 1:
      return int(ndig[0])
   return ndig

def can_pickle(thing):
   try:
      _pickle.dumps(thing)
      return True
   except:
      return False

def pickle_time(thing):
   t = time.perf_counter()
   _pickle.dumps(thing)
   return time.perf_counter() - t

def pickle_analysis(thing, mintime=0.1, loglevel='debug'):
   logme = getattr(log, loglevel.lower())
   logme('pickle_analysis:')
   for k, v in thing.items():
      if not can_pickle(v):
         logme(f'  cant pickle {k} : {v}')
      else:
         t = pickle_time(v)
         if t > mintime:
            logme(f'  pickle time of {k} is {t}'),

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

def load_threads(fnames, nthread=0):
   if nthread <= 0: nthread = cpu_count()
   with concurrent.futures.ThreadPoolExecutor(nthread) as exe:
      return list(exe.map(load, fnames))

class InProcessExecutor:
   def __init__(self, *args, **kw):
      pass

   def __enter__(self):
      return self

   def __exit__(self, *args):
      pass

   def submit(self, fn, *args, **kw):
      return NonFuture(fn, *args, **kw)

   # def map(self, func, *iterables):
   # return map(func, *iterables)
   # return (NonFuture(func(*args) for args in zip(iterables)))

class NonFuture:
   def __init__(self, fn, *args, dummy=None, **kw):
      self.fn = fn
      self.dummy = not callable(fn) if dummy is None else dummy
      self.args = args
      self.kw = kw
      self._condition = threading.Condition()
      self._state = "FINISHED"
      self._waiters = []

   def result(self):
      if self.dummy:
         return self.fn
      return self.fn(*self.args, **self.kw)
