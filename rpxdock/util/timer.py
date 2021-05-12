import time, os, collections, statistics, logging, numpy

log = logging.getLogger(__name__)

_summary_types = dict(
   sum=sum,
   mean=statistics.mean,
   min=min,
   max=max,
   median=statistics.median,
)

class _TimerGetter:
   def __init__(self, timer, summary):
      self.timer = timer
      self.summary = summary

   def __getattr__(self, name):
      if name in ('timer', 'checkpoints'):
         raise AttributeError
      if name in self.timer.checkpoints:
         return self.summary(self.timer.checkpoints[name])
      raise AttributeError("Timer has no attribute named: " + name)

   def __getitem__(self, name):
      return getattr(self, name)

class Timer:
   def __init__(self, name='Timer', verbose=False):
      self.name = name
      self.verbose = verbose
      self.sum = _TimerGetter(self, numpy.sum)
      self.mean = _TimerGetter(self, numpy.mean)
      self.min = _TimerGetter(self, numpy.min)
      self.max = _TimerGetter(self, numpy.max)
      self.median = _TimerGetter(self, numpy.median)

   def start(self):
      return self.__enter__()

   def stop(self):
      return self.__exit__()

   def __enter__(self):
      if self.verbose: log.debug(f'{self.name} intialized')
      self.start = time.perf_counter()
      self.last = self.start
      self.checkpoints = collections.defaultdict(list)
      return self

   def checkpoint(self, name='untracked', verbose=False):
      t = time.perf_counter()
      self.checkpoints[name].append(t - self.last)
      self.last = t
      if self.verbose or verbose:
         log.debug(f'{self.name} checkpoint {name} iter {len(self.checkpoints[name])}' +
                   f'time {self.checkpoints[name][-1]}')
      return self

   def __exit__(self, type=None, value=None, traceback=None):
      self.checkpoints['total'].append(time.perf_counter() - self.start)
      if self.verbose: log.debug(f'{self.name} finished')
      if self.verbose: self.report()
      return self

   def __getattr__(self, name):
      if name == "checkpoints":
         raise AttributeError
      if name in self.checkpoints:
         return self.checkpoints[name]
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

   def report(self, order='longest', summary='sum', namelen=None, precision='10.5f', printme=True,
              scale=1.0):
      if namelen is None:
         namelen = max(len(n) for n in self.checkpoints)
      lines = [f"Times(order={order}, summary={summary}):"]
      times = self.report_dict(order=order, summary=summary)
      for cpoint, t in times.items():
         lines.append(f'    {cpoint:>{namelen}} {t*scale:{precision}}')
         if scale == 1000: lines[-1] += 'ms'
      r = os.linesep.join(lines)
      if printme: log.info(r)
      return r

   @property
   def total(self):
      if 'total' in self.checkpoints:
         return sum(self.checkpoints['total'])
      return time.perf_counter() - self.start

   def __str__(self):
      return self.report(printme=False)

   def __repr__(self):
      return str(type(self))

   def merge(self, others):
      if isinstance(others, Timer): others = [others]
      for other in others:
         for k, v in other.checkpoints.items():
            self.checkpoints[k].extend(v)
