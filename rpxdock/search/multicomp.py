import itertools, functools, numpy as np, xarray as xr, rpxdock as rp, rpxdock.homog as hm
from rpxdock.search import hier_search, trim_ok

def make_multicomp(
      bodies,
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
   arg.output_prefix = arg.output_prefix if arg.output_prefix else spec.spec

   assert len(bodies) == spec.num_components
   bodies = list(bodies)
   if not fixed_components:
      for i, b in enumerate(bodies):
         bodies[i] = b.copy_xformed(rp.homog.align_vector([0, 0, 1], spec.axis[i]))

   dotrim = arg.max_trim and arg.trimmable_components and len(bodies) < 3
   Evaluator = MultiCompEvaluatorWithTrim if dotrim else MultiCompEvaluator
   evaluator = Evaluator(bodies, spec, hscore, **arg)
   xforms, scores, extra, stats = search(sampler, evaluator, **arg)
   ibest = rp.filter_redundancy(xforms, bodies, scores, **arg)
   tdump = _debug_dump_cage(xforms, bodies, spec, scores, ibest, evaluator, **arg)

   if arg.verbose:
      print(f"rate: {int(stats.ntot / t.total):,}/s ttot {t.total:7.3f} tdump {tdump:7.3f}")
      print("stage time:", " ".join([f"{t:8.2f}s" for t, n in stats.neval]))
      print("stage rate:  ", " ".join([f"{int(n/t):7,}/s" for t, n in stats.neval]))

   xforms = xforms[ibest]
   wrpx = arg.wts.sub(rpx=1, ncontact=0)
   wnct = arg.wts.sub(rpx=0, ncontact=1)
   rpx, extra = evaluator(xforms, arg.nresl - 1, wrpx)
   ncontact, *_ = evaluator(xforms, arg.nresl - 1, wnct)
   data = dict(
      attrs=dict(arg=arg, stats=stats, ttotal=t.total, tdump=tdump,
                 output_prefix=arg.output_prefix, output_body='all', sym=spec.spec),
      scores=(["model"], scores[ibest].astype("f4")),
      xforms=(["model", "comp", "hrow", "hcol"], xforms),
      rpx=(["model"], rpx.astype("f4")),
      ncontact=(["model"], ncontact.astype("f4")),
   )
   for k, v in extra.items():
      if not isinstance(v, (list, tuple)) or len(v) > 3:
         v = ['model'], v
      data[k] = v
   for i in range(len(bodies)):
      data[f'disp{i}'] = (['model'], np.sum(xforms[:, i, :3, 3] * spec.axis[None, i, :3], axis=1))
      data[f'angle{i}'] = (['model'], rp.homog.angle_of(xforms[:, i]) * 180 / np.pi)
   default_label = [f'comp{c}' for c in 'ABCDEFD' [:len(bodies)]]

   return rp.Result(
      body_=None if arg.dont_store_body_in_results else bodies,
      body_label_=[] if arg.dont_store_body_in_results else default_label,
      **data,
   )

class MultiCompEvaluatorBase:
   def __init__(self, bodies, spec, hscore, wts=rp.Bunch(ncontact=0.1, rpx=1.0), **kw):
      self.arg = rp.Bunch(kw)
      self.hscore = hscore
      self.symrots = [rp.geom.symframes(n) for n in spec.nfold]
      self.spec = spec
      self.arg.wts = wts
      self.bodies = [b.copy_with_sym(spec.nfold[i], spec.axis[i]) for i, b in enumerate(bodies)]

class MultiCompEvaluator(MultiCompEvaluatorBase):
   def __init__(self, *arg, **kw):
      super().__init__(*arg, **kw)

   def __call__(self, xforms, iresl=-1, wts={}, **kw):
      arg = self.arg.sub(wts=wts)
      xeye = np.eye(4, dtype="f4")
      bod = self.bodies
      xforms = xforms.reshape(-1, xforms.shape[-3], 4, 4)
      xnbr = self.spec.to_neighbor_olig

      # check for "flatness"
      delta_h = np.array(
         [hm.hdot(xforms[:, i] @ bod[i].com(), self.spec.axis[i]) for i in range(len(bod))])
      ok = np.max(np.abs(delta_h[None] - delta_h[:, None]), axis=(0, 1)) < arg.max_delta_h
      # ok = np.repeat(True, len(xforms))

      # check clash, or get non-clash range
      for i in range(len(bod)):
         if xnbr[i] is not None:
            ok[ok] &= bod[i].clash_ok(bod[i], xforms[ok, i], xnbr[i] @ xforms[ok, i], **arg)
         for j in range(i):
            ok[ok] &= bod[i].clash_ok(bod[j], xforms[ok, i], xforms[ok, j], **arg)

      if xnbr[0] is None and xnbr[1] is not None and xnbr[2] is not None:  # layer hack
         inv = np.linalg.inv
         ok[ok] &= bod[0].clash_ok(bod[1], xforms[ok, 0], xnbr[1] @ xforms[ok, 1], **arg)
         ok[ok] &= bod[0].clash_ok(bod[2], xforms[ok, 0], xnbr[2] @ xforms[ok, 2], **arg)
         ok[ok] &= bod[0].clash_ok(bod[1], xforms[ok, 0], xnbr[2] @ xforms[ok, 1], **arg)
         ok[ok] &= bod[0].clash_ok(bod[2], xforms[ok, 0], xnbr[1] @ xforms[ok, 2], **arg)
         ok[ok] &= bod[1].clash_ok(bod[2], xforms[ok, 1], xnbr[2] @ xforms[ok, 2], **arg)
         ok[ok] &= bod[1].clash_ok(bod[2], xforms[ok, 1], xnbr[1] @ xforms[ok, 2], **arg)
         ok[ok] &= bod[0].clash_ok(bod[1], xforms[ok, 0], inv(xnbr[1]) @ xforms[ok, 1], **arg)
         # ok[ok] &= bod[0].clash_ok(bod[2], xforms[ok, 0], inv(xnbr[2]) @ xforms[ok, 2], **arg)
         # ok[ok] &= bod[1].clash_ok(bod[2], xforms[ok, 1], inv(xnbr[2]) @ xforms[ok, 2], **arg)
         ok[ok] &= bod[0].clash_ok(bod[2], xforms[ok, 0], inv(xnbr[1]) @ xforms[ok, 2], **arg)
         ok[ok] &= bod[1].clash_ok(bod[2], xforms[ok, 1], inv(xnbr[1]) @ xforms[ok, 2], **arg)

      # score everything that didn't clash
      ifscore = list()
      for i in range(len(bod)):
         for j in range(i):
            ifscore.append(
               self.hscore.scorepos(bod[j], bod[i], xforms[ok, j], xforms[ok, i], iresl, wts=wts))
      # ifscore = np.stack(ifscore)
      # print(ifscore.shape)
      scores = np.zeros(len(xforms))
      scores[ok] = arg.iface_summary(ifscore, axis=0)

      # bod[0].pos = xforms[np.argmax(scores), 0]
      # bod[1].pos = xforms[np.argmax(scores), 1]
      # bod[0].dump_pdb('test0.pdb')
      # bod[1].dump_pdb('test1.pdb')
      # assert 0

      return scores, rp.Bunch()

class MultiCompEvaluatorWithTrim(MultiCompEvaluatorBase):
   def __init__(self, *arg, trimmable_components="AB", **kw):
      super().__init__(*arg, **kw)
      self.trimmable_components = trimmable_components

   def __call__(self, *arg, **kw):
      if self.trimmable_components.upper() in ("AB", "BA"):
         sa, lba, uba = self.eval_trim_one('A', *arg, **kw)
         sb, lbb, ubb = self.eval_trim_one('B', *arg, **kw)
         ia = (sa > sb).reshape(-1, 1)
         lb = np.where(ia, lba, lbb)
         ub = np.where(ia, uba, ubb)
         scores = np.maximum(sa, sb)
      else:
         assert self.trimmable_components.upper() in "AB"
         scores, lb, ub = self.eval_trim_one(self.trimmable_components, *arg, **kw)

      extra = rp.Bunch(reslb=(['model', 'component'], lb), resub=(['model', 'component'], ub))
      return scores, extra

   def eval_trim_one(self, trim_component, x, iresl=-1, wts={}, **kw):
      arg = self.arg.sub(wts=wts)
      compA, compB = self.bodies
      x = x.reshape(-1, 2, 4, 4)
      xnbr = self.spec.to_neighbor_olig

      # check for "flatness"
      d1 = hm.hdot(x[:, 0] @ compA.com(), self.spec.axis[0])
      d2 = hm.hdot(x[:, 1] @ compB.com(), self.spec.axis[1])
      ok = abs(d1 - d2) < arg.max_delta_h

      lbA = np.zeros(len(x), dtype='i4')
      lbB = np.zeros(len(x), dtype='i4')
      ubA = np.ones(len(x), dtype='i4') * (self.bodies[0].asym_body.nres - 1)
      ubB = np.ones(len(x), dtype='i4') * (self.bodies[1].asym_body.nres - 1)

      if trim_component == 'A':
         trimA1 = compA.intersect_range(compB, x[ok, 0], x[ok, 1], **arg)
         trimA1, trimok = trim_ok(trimA1, compA.asym_body.nres, **arg)
         ok[ok] &= trimok

         xa = x[ok, 0]
         if xnbr is not None:
            trimA2 = compA.intersect_range(compA, xa, xnbr[0] @ xa, **arg)
         trimA2, trimok2 = trim_ok(trimA2, compA.asym_body.nres, **arg)
         ok[ok] &= trimok2
         lbA[ok] = np.maximum(trimA1[0][trimok2], trimA2[0])
         ubA[ok] = np.minimum(trimA1[1][trimok2], trimA2[1])

         xb = x[ok, 1]
         if xnbr is not None:
            trimB = compB.intersect_range(compB, xb, xnbr[1] @ xb, **arg)
         trimB, trimok = trim_ok(trimB, compB.asym_body.nres, **arg)
         ok[ok] &= trimok
         lbB[ok], ubB[ok] = trimB
      elif trim_component == 'B':
         trimB1 = compB.intersect_range(compA, x[ok, 1], x[ok, 0], **arg)
         trimB1, trimok = trim_ok(trimB1, compB.asym_body.nres, **arg)
         ok[ok] &= trimok

         xb = x[ok, 1]
         if xnbr is not None:
            trimB2 = compB.intersect_range(compB, xb, xnbr[1] @ xb, **arg)
         trimB2, trimok2 = trim_ok(trimB2, compB.asym_body.nres, **arg)
         ok[ok] &= trimok2
         lbB[ok] = np.maximum(trimB1[0][trimok2], trimB2[0])
         ubB[ok] = np.minimum(trimB1[1][trimok2], trimB2[1])

         xa = x[ok, 0]
         if xnbr is not None:
            trimA = compA.intersect_range(compA, xa, xnbr[0] @ xa, **arg)
         trimA, trimok = trim_ok(trimA, compA.asym_body.nres, **arg)
         ok[ok] &= trimok
         lbA[ok], ubA[ok] = trimA
      else:
         raise ValueError('trim_component invalid')

      # score everything that didn't clash
      bounds = lbA[ok], ubA[ok], compA.asym_body.nres, lbB[ok], ubB[ok], compB.asym_body.nres
      scores = np.zeros(len(x))
      scores[ok] = self.hscore.scorepos(body1=compA, body2=compB, pos1=x[ok, 0], pos2=x[ok, 1],
                                        iresl=iresl, bounds=bounds, **arg)

      # if np.sum(ok):
      # print(iresl, np.sum(ok), np.min(scores[ok]), np.max(scores[ok]), np.mean(lbA),
      # np.mean(ubA), np.mean(lbB), np.mean(ubB))

      return scores, np.stack([lbA, lbB], axis=1), np.stack([ubA, ubB], axis=1)

def _debug_dump_cage(xforms, bodies, spec, scores, ibest, evaluator, **kw):
   arg = rp.Bunch(kw)
   t = rp.Timer().start()
   nout_debug = min(10 if arg.nout_debug is None else arg.nout_debug, len(ibest))
   for iout in range(nout_debug):
      i = ibest[iout]
      bodies[0].move_to(xforms[i, 0])
      bodies[1].move_to(xforms[i, 1])
      wrpx, wnct = (arg.wts.sub(rpx=1, ncontact=0), arg.wts.sub(rpx=0, ncontact=1))
      scr, extra = evaluator(xforms[i], arg.nresl - 1, wrpx)
      cnt, extra = evaluator(xforms[i], arg.nresl - 1, wnct)
      fn = arg.output_prefix + "_%02i.pdb" % iout
      lbub = [extra.lbub] if extra.lbub else []
      if len(lbub) > 1:
         print(
            f"{fn} score {scores[i]:7.3f} rpx {scr[0]:7.3f} cnt {cnt[0]:4}",
            f"resi {lbub[0][0]}-{lbub[1][0]}",
         )
      else:
         print(f"{fn} score {scores[i]:7.3f} rpx {scr[0]:7.3f} cnt {cnt[0]:4}")
      rp.dump_pdb_from_bodies(fn, bodies, rp.geom.symframes(spec.sym, xforms[iout]),
                              resbounds=lbub)
   return t.total
