import _pickle, os, multiprocessing, threading
import numpy as np

def load(f):
   if isinstance(f, str):
      print('loading', f)
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

class ThreadLoader(threading.Thread):
   def __init__(self, fname):
      super().__init__(None, None, None)
      self.fname = fname

   def run(self):
      self.result = load(self.fname)
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
