import time, os, collections

class Timer:
   def __enter__(self):
      self.start = time.perf_counter()
      self.last = self.start
      self.checkpoints = collections.defaultdict(lambda: 0)
      return self

   def checkpoint(self, name='none'):
      t = time.perf_counter()
      self.checkpoints[name] += t - self.last
      self.last = t

   def __exit__(self, type, value, traceback):
      self.checkpoint()
      self.checkpoints['total'] = time.perf_counter() - self.start

   def __getattr__(self, name):
      if name in self.checkpoints:
         return self.checkpoints[name]
      raise AttributeError("Timer has no attribute named: " + name)

   def report_dict(self, order='longest'):
      if order == 'longest':
         reordered = sorted(self.checkpoints.items(), key=lambda kv: -kv[1])
         return {k: v for k, v in reordered}
      elif order == 'callorder':
         return self.checkpoints
      else:
         raise ValueError('Timer, unknown order: ' + order)

   def report(self, order='longest', namelen=None, precision='10.5f'):
      if namelen is None:
         namelen = max(len(n) for n in self.checkpoints)
      lines = [f"Times(order={order}):"]
      times = self.report_dict(order=order)
      for cpoint, t in times.items():
         lines.append(f'    {cpoint:>{namelen}} {t:{precision}}')
      return os.linesep.join(lines)

   def __str__(self):
      return self.report()
