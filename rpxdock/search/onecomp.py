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
   arg = rp.Bunch(kw)
   t = rp.Timer().start()
   arg.nresl = len(hscore.hier) if arg.nresl is None else arg.nresl
   arg.output_prefix = arg.output_prefix if arg.output_prefix else spec.arch

   assert isinstance(body, rp.Body)
   if not fixed_components:
      body = body.copy_xformed(rp.homog.align_vector([0, 0, 1], spec.axis))

   dotrim = arg.max_trim and arg.trimmable_components
   evaluator = OneCompEvaluator(body, spec, hscore, **arg)
   xforms, scores, extra, stats = search(sampler, evaluator, **arg)
   ibest = rp.filter_redundancy(xforms, body, scores, **arg)
   # tdump = _debug_dump_cage(xforms, body, spec, scores, ibest, evaluator, **arg)

   if arg.verbose:
      print(f"rate: {int(stats.ntot / t.total):,}/s ttot {t.total:7.3f}")
      print("stage time:", " ".join([f"{t:8.2f}s" for t, n in stats.neval]))
      print("stage rate:  ", " ".join([f"{int(n/t):7,}/s" for t, n in stats.neval]))

   xforms = xforms[ibest]
   wrpx = arg.wts.sub(rpx=1, ncontact=0)
   wnct = arg.wts.sub(rpx=0, ncontact=1)
   rpx, extra = evaluator(xforms, arg.nresl - 1, wrpx)
   ncontact, *_ = evaluator(xforms, arg.nresl - 1, wnct)
   data = dict(
      attrs=dict(arg=arg, stats=stats, ttotal=t.total, output_prefix=arg.output_prefix,
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
   data['disp'] = (['model'], np.sum(xforms[:, :3, 3] * spec.axis[None, :3], axis=1))
   data['angle'] = (['model'], rp.homog.angle_of(xforms[:]) * 180 / np.pi)
   default_label = ['compA']

   return rp.Result(
      body_=None if arg.dont_store_body_in_results else [body],
      body_label_=[] if arg.dont_store_body_in_results else default_label,
      **data,
   )

class OneCompEvaluator:
   def __init__(self, body, spec, hscore, wts=rp.Bunch(ncontact=0.1, rpx=1.0),
                trimmable_components="AB", **kw):
      self.arg = rp.Bunch(kw)
      self.hscore = hscore
      self.symrot = rp.geom.symframes(spec.nfold)
      self.spec = spec
      self.arg.wts = wts
      self.body = body.copy_with_sym(spec.nfold, spec.axis)
      self.trimmable_components = trimmable_components

   def __call__(self, xforms, iresl=-1, wts={}, **kw):
      arg = self.arg.sub(wts=wts)
      xeye = np.eye(4, dtype="f4")
      body, sfxn = self.body, self.hscore.scorepos
      X = xforms.reshape(-1, 4, 4)  #@ body.pos
      Xsym = self.spec.to_neighbor_olig @ X

      # check clash, or get non-clash range
      ok = np.ones(len(xforms), dtype='bool')
      if arg.max_trim > 0:
         trim = body.intersect_range(body, X[ok], Xsym[ok], **arg)
         trim, trimok = rp.search.trim_ok(trim, body.nres, **arg)
         ok[ok] &= trimok
      else:
         ok[ok] &= body.clash_ok(body, X[ok], Xsym[ok], **arg)
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
      scores[ok] = sfxn(body, body, X[ok], Xsym[ok], iresl, bounds, **arg)

      # record ranges used
      lb = np.zeros(len(scores), dtype="i4")
      ub = np.ones(len(scores), dtype="i4") * (body.nres - 1)
      if trim: lb[ok], ub[ok] = trim[0], trim[1]

      return scores, rp.Bunch(reslb=lb, resub=ub)
