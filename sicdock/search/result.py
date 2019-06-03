import numpy as np, xarray as xr

def concat_results(results, attrs=dict(), datasetkey="dataset"):
   for k in results[0]:
      if k != datasetkey:
         for r in results:
            ds = r[datasetkey]
            try:
               ds[k] = (["model"], r[k])
            except ValueError:
               ds[k] = (["model"], np.repeat(r[k], len(ds.model)))
   d = xr.concat([r[datasetkey] for r in results], "model")
   d.attrs.update(attrs)
   return d
