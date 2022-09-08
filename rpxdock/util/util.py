import _pickle, os, multiprocessing, threading, copy
import hashlib, logging, concurrent, time, gzip, bz2, lzma, zipfile, json
from collections import abc
import numpy as np
from willutil import Bunch
from xarray.backends import netCDF4_

log = logging.getLogger(__name__)

def load(f, verbose=True):
   from rpxdock.motif import respairscore_from_tarball, xmap_from_tarball
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
      elif f.endswith('.xmap.txz'):
         return xmap_from_tarball(f)
      elif f.endswith('.rpx.txz'):
         return respairscore_from_tarball(f)
      elif f.endswith('.nc'):
         import xarray as xr
         return xr.load_dataset(f)
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

def sanitize_for_storage(data, netcdf=False, _n=0):
   import xarray as xr
   newdata = copy.copy(data)

   if isinstance(data, (np.ndarray, xr.Dataset, xr.DataArray, int, float, str)):
      pass
   elif isinstance(data, abc.MutableMapping):
      for k, v in data.items():
         newdata[k] = sanitize_for_storage(v, netcdf=netcdf, _n=_n + 1)
      if netcdf:
         newdata = list(newdata.items())
   elif isinstance(data, abc.MutableSequence):
      for i, v in enumerate(data):
         newdata[i] = sanitize_for_storage(v, netcdf=netcdf, _n=_n + 1)
   elif isinstance(data, tuple):
      newdata = tuple(sanitize_for_storage(list(data), netcdf=netcdf))
   elif isinstance(data, abc.Set):
      newdata = data.__class__(sanitize_for_storage(list(data), netcdf=netcdf, _n=_n + 1))
      if netcdf:
         newdata = list(newdata)
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
      newdata = m + '.' + n

   return newdata

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

def check_eq_json(a, b):
   return json.loads(json.dumps(a)) == json.loads(json.dumps(b))
