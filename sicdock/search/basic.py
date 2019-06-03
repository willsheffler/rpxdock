from time import perf_counter
from concurrent.futures import ThreadPoolExecutor
import numpy as np

def grid_search(sampler, evaluator, **kw):
   if (not isinstance(sampler, np.ndarray) or sampler.ndim != 3
       or sampler.shape[-2:] != (4, 4)):
      raise ValueError('sampler for grid_search should be array of samples')
   xforms = sampler
   scores, *resbound, t = evaluate_positions(evaluator, xforms, **kw)
   stats = Bunch(ntot=len(xforms), neval=[len(xforms)])
   return xforms, scores, stats

def evaluate_positions(evaluator, executor=None, **kw):
   t = perf_counter()
   if executor:
      return (*evaluate_positions_executor(executor, evaluator, **kw), perf_counter() - t)
   iface_scores, lb, ub = evaluator(**kw)
   return iface_scores, lb, lb, perf_counter() - t

def evaluate_positions_executor(executor, evaluator, xforms, **kw):
   assert isinstance(executor, ThreadPoolExecutor)
   nworkers = executor._max_workers
   assert nworkers > 0
   ntasks = int(len(xforms) / 10000)
   ntasks = max(nworkers, ntasks)
   futures = list()
   for i, x in enumerate(np.array_split(xforms, ntasks)):
      futures.append(executor.submit(evaluator, x, **kw))
      futures[-1].idx = i
   # futures = [f for f in as_completed(futures)]
   results = [f.result() for f in sorted(futures, key=lambda x: x.idx)]
   iface_scores = np.concatenate([r[0] for r in results])
   lb = np.concatenate([r[1] for r in results])
   ub = np.concatenate([r[2] for r in results])
   return iface_scores, lb, ub

def trim_atom_to_res_numbering(ptrim, nres, max_trim):
   ptrim = ((ptrim[0] - 1) // 5 + 1, (ptrim[1] + 1) // 5 - 1)  # to res numbers
   ntrim = ptrim[0] + nres - ptrim[1] - 1
   trimok = ntrim <= max_trim
   ptrim = (ptrim[0][trimok], ptrim[1][trimok])
   return ptrim, trimok
