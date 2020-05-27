import itertools, functools, numpy as np, xarray as xr, rpxdock as rp, rpxdock.homog as hm
from rpxdock.search import hier_search, trim_ok

def make_onecomp(
   body,
   spec,
   hscore,
   search=hier_search,
   sampler=None,
   fixed_components=False,
   **kw,
):
   kw = rp.Bunch(kw)
   t = rp.Timer().start()
   kw.nresl = hscore.actual_nresl if kw.nresl is None else kw.nresl
   kw.output_prefix = kw.output_prefix if kw.output_prefix else spec.arch

   assert isinstance(body, rp.Body)
   if not fixed_components:
      body = body.copy_xformed(rp.homog.align_vector([0, 0, 1], spec.axis))

   dotrim = kw.max_trim and kw.trimmable_components
   evaluator = OneCompEvaluator(body, spec, hscore, **kw)
   xforms, scores, extra, stats = search(sampler, evaluator, **kw)
   ibest = rp.filter_redundancy(xforms, body, scores, **kw)
   # tdump = _debug_dump_cage(xforms, body, spec, scores, ibest, evaluator, **kw)

   if kw.verbose:
      print(f"rate: {int(stats.ntot / t.total):,}/s ttot {t.total:7.3f}")
      print("stage time:", " ".join([f"{t:8.2f}s" for t, n in stats.neval]))
      print("stage rate:  ", " ".join([f"{int(n/t):7,}/s" for t, n in stats.neval]))

   xforms = xforms[ibest]
   wrpx = kw.wts.sub(rpx=1, ncontact=0)
   wnct = kw.wts.sub(rpx=0, ncontact=1)
   rpx, extra = evaluator(xforms, kw.nresl - 1, wrpx)
   ncontact, *_ = evaluator(xforms, kw.nresl - 1, wnct)
   data = dict(
      attrs=dict(arg=kw, stats=stats, ttotal=t.total, output_prefix=kw.output_prefix,
                 output_body='all', sym=spec.arch),
      scores=(["model"], scores[ibest].astype("f4")),
      xforms=(["model", "hrow", "hcol"], xforms),
      rpx=(["model"], rpx.astype("f4")),
      ncontact=(["model"], ncontact.astype("f4")),
   )
   for k, v in extra.items():
      if not isinstance(v, (list, tuple)) or len(v) > 3:
         v = ['model'], v
      data[k] = v
   data['disp'] = (['model'], np.sum(xforms[:, :3, 3] * spec.axis[None, :3], axis=-1).squeeze())
   data['angle'] = (['model'], rp.homog.angle_of(xforms[:]) * 180 / np.pi)
   default_label = ['compA']

   return rp.Result(
      body_=None if kw.dont_store_body_in_results else [body],
      body_label_=[] if kw.dont_store_body_in_results else default_label,
      **data,
   )

class OneCompEvaluator:
   def __init__(self, body, spec, hscore, wts=rp.Bunch(ncontact=0.1, rpx=1.0),
                trimmable_components="AB", **kw):
      self.kw = rp.Bunch(kw)
      self.hscore = hscore
      self.symrot = rp.geom.symframes(spec.nfold)
      self.spec = spec
      self.kw.wts = wts
      self.body = body.copy_with_sym(spec.nfold, spec.axis)
      self.trimmable_components = trimmable_components

   def __call__(self, xforms, iresl=-1, wts={}, **kw):
      kw = self.kw.sub(wts=wts)
      xeye = np.eye(4, dtype="f4")
      body, sfxn = self.body, self.hscore.scorepos
      X = xforms.reshape(-1, 4, 4)  #@ body.pos
      Xsym = self.spec.to_neighbor_olig @ X

      # check clash, or get non-clash range
      ok = np.ones(len(xforms), dtype='bool')
      if kw.max_trim > 0:
         trim = body.intersect_range(body, X[ok], Xsym[ok], **kw)
         trim, trimok = rp.search.trim_ok(trim, body.nres, **kw)
         ok[ok] &= trimok
      else:
         ok[ok] &= body.clash_ok(body, X[ok], Xsym[ok], **kw)
         trim = [0], [body.nres - 1]

      # if iresl == 4:
      #    i = 0
      #    body.pos = X[i]
      #    body.dump_pdb('topscore1.pdb')
      #    body.pos = Xsym[i]
      #    body.dump_pdb('topscore2.pdb')
      #    assert 0

      # score everything that didn't clash
      scores = np.zeros(len(X))
      bounds = (*trim, -1, *trim, -1)
      scores[ok] = sfxn(body, body, X[ok], Xsym[ok], iresl, bounds, **kw)

      # record ranges used
      lb = np.zeros(len(scores), dtype="i4")
      ub = np.ones(len(scores), dtype="i4") * (body.nres - 1)
      if trim: lb[ok], ub[ok] = trim[0], trim[1]

      return scores, rp.Bunch(reslb=lb, resub=ub)
