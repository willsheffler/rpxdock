from collections import OrderedDict
import numpy as np
import xarray as xr
from sicdock.util import sanitize_for_pickle
from sicdock.body import Body

class Result:
   def __init__(self, data_or_file=None, body_=[], **kw):
      if isinstance(body_, Body): body_ = [body_]
      self.bodies = [body_]
      if data_or_file:
         assert len(kw) is 0
         if isinstance(data_or_file, xr.Dataset):
            self.data = data_or_file
         else:
            self.load(file_)
      else:
         attrs = OrderedDict(kw['attrs']) if 'attrs' in kw else None
         if attrs: del kw['attrs']
         attrs = sanitize_for_pickle(attrs)
         self.data = xr.Dataset(dict(**kw), attrs=attrs)

   def __getattr__(self, name):
      if name == "data":
         raise AttributeError
      return getattr(self.data, name)

   def __getitem__(self, name):
      return self.data[name]

   def __setitem__(self, name, val):
      self.data[name] = val

   def __str__(self):
      return "Result with data = " + str(self.data).replace("\n", "\n  ")

   def copy(self):
      return Result(self.data.copy())

   def getstate(self):
      return self.data.to_dict()

   def setstate(self, state):
      self.data = xr.Dataset.from_dict(state)

   def __len__(self):
      return len(self.model)

   def __eq__(self, other):
      return self.data == other.data

def concat_results(results, **kw):
   imeta = np.repeat(np.arange(len(results)), [len(r) for r in results])
   assert max(len(r.bodies) for r in results) == 1
   r = Result(xr.concat([r.data for r in results], dim='model', **kw))
   r.bodies = [r.bodies[0] for r in results]
   r.data['imeta'] = (['model'], imeta)
   r.data.attrs = OrderedDict(meta=[r.attrs for r in results])
   return r

def dummy_result(size=1000):
   from sicdock.homog import rand_xform
   return Result(
      scores=(["model"], np.random.rand(size).astype('f4')),
      xforms=(["model", "hrow", "hcol"], rand_xform(size).astype('f4')),
      rpx_plug=(["model"], np.random.rand(size).astype('f4')),
      rpx_hole=(["model"], np.random.rand(size).astype('f4')),
      ncontact_plug=(["model"], np.random.rand(size).astype('f4')),
      ncontact_hole=(["model"], np.random.rand(size).astype('f4')),
      reslb=(["model"], np.random.randint(0, 100, size)),
      resub=(["model"], np.random.randint(100, 200, size)),
   )