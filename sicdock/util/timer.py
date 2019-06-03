import time, os, collections, statistics

_summary_types = dict(
   sum=sum,
   mean=statistics.mean,
   min=min,
   max=max,
   median=statistics.median,
)

class Timer:
   def __init__(self, name='Timer', verbose=False):
      self.name = name
      self.verbose = verbose

   def start(self):
      return self.__enter__()

   def stop(self):
      return self.__exit__()

   def __enter__(self):
      if self.verbose: print(f'{self.name} intialized')
      self.start = time.perf_counter()
      self.last = self.start
      self.checkpoints = collections.defaultdict(list)
      return self

   def checkpoint(self, name='none', verbose=False):
      if self.verbose or verbose:
         print(f'{self.name} checkpoint {name}", "iter {len(self.checkpoints[name])}')
      t = time.perf_counter()
      self.checkpoints[name].append(t - self.last)
      self.last = t
      return self

   def __exit__(self, type=None, value=None, traceback=None):
      self.checkpoints['total'].append(time.perf_counter() - self.start)
      if self.verbose: print(f'{self.name} finished')
      if self.verbose: self.report()
      return self

   def __getattr__(self, name):
      if name in self.checkpoints:
         return sum(self.checkpoints[name])
      raise AttributeError("Timer has no attribute named: " + name)

   def alltimes(self, name):
      return self.checkpoints[name]

   def report_dict(self, order='longest', summary='sum'):
      if not callable(summary):
         if summary not in _summary_types:
            raise ValueError('unknown summary type: ' + str(summary))
         summary = _summary_types[summary]
      if order == 'longest':
         reordered = sorted(self.checkpoints.items(), key=lambda kv: -summary(kv[1]))
         return {k: summary(v) for k, v in reordered}
      elif order == 'callorder':
         return self.checkpoints
      else:
         raise ValueError('Timer, unknown order: ' + order)

   def report(self, order='longest', summary='sum', namelen=None, precision='10.5f', printme=True):
      if namelen is None:
         namelen = max(len(n) for n in self.checkpoints)
      lines = [f"Times(order={order}, summary={summary}):"]
      times = self.report_dict(order=order, summary=summary)
      for cpoint, t in times.items():
         lines.append(f'    {cpoint:>{namelen}} {t:{precision}}')
      r = os.linesep.join(lines)
      if printme: print(r)
      return r

   def __str__(self):
      return self.report(printme=False)
