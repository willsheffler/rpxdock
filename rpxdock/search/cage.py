import itertools, numpy as np, xarray as xr, rpxdock as rp, rpxdock.homog as hm
from rpxdock.search import hier_search, trim_atom_to_res_numbering

def hier_cage_sampler(spec, lb=25, ub=200, resl=10, angresl=10, flip1=True, flip2=True):
   cart_nstep = int(np.ceil((ub - lb) / resl))
   ang1, ang2 = 360 / spec.nfold1, 360 / spec.nfold2
   ang1_nstep = int(np.ceil(ang1 / angresl))
   ang2_nstep = int(np.ceil(ang2 / angresl))

   samp1 = rp.sampling.RotCart1Hier_f4(lb, ub, cart_nstep, 0, ang1, ang1_nstep, spec.axis1[:3])
   samp2 = rp.sampling.RotCart1Hier_f4(lb, ub, cart_nstep, 0, ang2, ang2_nstep, spec.axis2[:3])
   if flip1:
      flip1 = rp.GridHier([np.eye(4), spec.xflip[0]])
      samp1 = rp.ProductHier(samp1, flip1)
   if flip2:
      flip2 = rp.GridHier([np.eye(4), spec.xflip[1]])
      samp2 = rp.ProductHier(samp2, flip2)

   return rp.CompoundHier(samp1, samp2)

def make_cage(bodies, spec, hscore, search=hier_search, sampler=None, **kw):
   arg = rp.Bunch(kw)
   t = rp.Timer().start()
   arg.nresl = len(hscore.hier) if arg.nresl is None else arg.nresl
   arg.output_prefix = arg.output_prefix if arg.output_prefix else spec.spec

   Evaluator = CageEvaluatorTrim if arg.max_trim else CageEvaluatorNoTrim
   evaluator = Evaluator(bodies, spec, hscore, **arg)
   xforms, scores, stats = search(sampler, evaluator, **arg)
   ibest = rp.filter_redundancy(xforms, bodies, scores, **arg)
   tdump = dump_cage(xforms, bodies, spec, scores, ibest, evaluator, **arg)

   if arg.verbose:
      print(f"rate: {int(stats.ntot / t.total):,}/s ttot {t.total:7.3f} tdump {tdump:7.3f}")
      print("stage time:", " ".join([f"{t:8.2f}s" for t, n in stats.neval]))
      print("stage rate:  ", " ".join([f"{int(n/t):7,}/s" for t, n in stats.neval]))

   xforms = xforms[ibest]
   wrpx = arg.wts.sub(rpx=1, ncontact=0)
   wnct = arg.wts.sub(rpx=0, ncontact=1)
   rpx, lb, ub = evaluator(xforms, arg.nresl - 1, wrpx)
   ncontact, *_ = evaluator(xforms, arg.nresl - 1, wnct)
   data = dict(
      attrs=dict(arg=arg, stats=stats, ttotal=t.total, tdump=tdump,
                 output_prefix=arg.output_prefix, output_body='all', sym=spec.spec),
      scores=(["model"], scores[ibest].astype("f4")),
      xforms=(["model", "comp", "hrow", "hcol"], xforms),
      rpx=(["model"], rpx.astype("f4")),
      ncontact=(["model"], ncontact.astype("f4")),
      reslb=(["model"], lb),
      resub=(["model"], ub),
   )
   for i in range(len(bodies)):
      data[f'disp{i}'] = (['model'], np.sum(xforms[:, i, :3, 3] * spec.axis[None, i, :3], axis=1))
      data[f'angle{i}'] = (['model'], rp.homog.angle_of(xforms[:, i]) * 180 / np.pi)
   return rp.Result(
      body_=None if arg.dont_store_body_in_results else bodies,
      body_label_=[] if arg.dont_store_body_in_results else ['comp1', 'comp2'],
      **data,
   )

class EvaluatorBase:
   def __init__(self, bodies, spec, hscore, max_delta_h=9e9, clashdis=3.5, max_trim=100,
                wts=rp.Bunch(ncontact=0.1, rpx=1.0), **kw):
      self.arg = rp.Bunch(kw)
      self.hscore = hscore
      self.symrots = [rp.geom.symframes(n) for n in spec.nfold]
      self.bodies = bodies
      self.spec = spec
      self.max_delta_h = max_delta_h
      self.clashdis = clashdis
      self.max_trim = max_trim
      self.wts = wts

class CageEvaluatorTrim(EvaluatorBase):
   def __init__(self, *args, max_trim=100, **kw):
      super().__init__(*args, **kw)

   def __call__(self, xforms, iresl=-1, wts={}, **kw):
      wts = self.wts.sub(wts)
      xeye = np.eye(4, dtype="f4")
      compA, compB = self.bodies
      xforms = xforms.reshape(-1, 2, 4, 4)

      # check for "flatness"
      d1 = hm.hdot(xforms[:, 0] @ compA.com(), self.spec.axis[0])
      d2 = hm.hdot(xforms[:, 1] @ compB.com(), self.spec.axis[1])
      ok = abs(d1 - d2) < self.max_delta_h

      # check clash, or get non-clash range
      plb = np.zeros(len(ok), dtype='i4')
      pub = np.ones(len(ok), dtype='i4') * (compA.nres - 1)
      for i, (frameA, frameB) in enumerate(itertools.product(*self.spec.compframes)):
         xformsA = xforms[ok, 0] @ frameA @ compA.pos
         xformsB = xforms[ok, 1] @ frameB @ compA.pos
         ptrim0 = compA.intersect_range(compB, self.clashdis, self.max_trim, xformsA, xformsB)
         ptrim0, trimok = trim_atom_to_res_numbering(ptrim0, compA.nres, self.max_trim)
         ok[ok] &= trimok
         plb[ok] = np.maximum(plb[ok], ptrim0[0])
         pub[ok] = np.minimum(pub[ok], ptrim0[1])
      print('nplb', np.sum(plb > 0))
      print('npub', np.sum(pub < compA.nres - 1))

      # score everything that didn't clash
      scores = np.zeros(len(xforms))
      for frameA, frameB in zip(*self.spec.compframes):
         xformsA = xforms[ok, 0] @ frameA @ compA.pos
         xformsB = xforms[ok, 1] @ frameB @ compB.pos
         scores[ok] += self.hscore.scorepos(compA, compB, xformsA, xformsB, wts, iresl,
                                            bounds=[plb[ok], pub[ok]])
      return scores, plb, pub

class CageEvaluatorNoTrim(CageEvaluatorTrim):
   def __init__(self, *args, **kw):
      super().__init__(*args, **kw)
      self.bodies[0] = self.bodies[0].copy_with_sym(self.spec.nfold1, self.spec.axis1)
      self.bodies[1] = self.bodies[1].copy_with_sym(self.spec.nfold2, self.spec.axis2)

   def __call__(self, xforms, iresl=-1, wts={}, **kw):
      wts = self.wts.sub(wts)
      xeye = np.eye(4, dtype="f4")
      compA, compB = self.bodies
      xforms = xforms.reshape(-1, 2, 4, 4)

      # # check for "flatness"
      d1 = hm.hdot(xforms[:, 0] @ compA.com(), self.spec.axis[0])
      d2 = hm.hdot(xforms[:, 1] @ compB.com(), self.spec.axis[1])
      ok = abs(d1 - d2) < self.max_delta_h

      # check clash, or get non-clash range
      xnbr = self.spec.to_neighbor_olig
      x0, x1 = xforms[:, 0] @ compA.pos, xforms[:, 1] @ compB.pos
      ok[ok] &= compA.clash_ok(compA, self.clashdis, x0[ok], xnbr[0] @ x0[ok])
      ok[ok] &= compB.clash_ok(compB, self.clashdis, x1[ok], xnbr[1] @ x1[ok])
      ok[ok] &= compA.clash_ok(compB, self.clashdis, x0[ok], x1[ok])

      # score everything that didn't clash
      scores = np.zeros(len(xforms))
      scores[ok] += self.hscore.scorepos(compA, compB, xforms[ok, 0], xforms[ok, 1], wts, iresl)

      plb = np.zeros(len(scores), dtype="i4")
      pub = np.ones(len(scores), dtype="i4") * (compA.nres - 1)
      return scores, plb, pub

def dump_cage(xforms, bodies, spec, scores, ibest, evaluator, **kw):
   arg = rp.Bunch(kw)
   t = rp.Timer().start()
   nout_debug = min(10 if arg.nout_debug is None else arg.nout_debug, len(ibest))
   for iout in range(nout_debug):
      i = ibest[iout]
      bodies[0].move_to(xforms[i, 0])
      bodies[1].move_to(xforms[i, 1])
      wrpx, wnct = (arg.wts.sub(rpx=1, ncontact=0), arg.wts.sub(rpx=0, ncontact=1))
      scr, *lbub = evaluator(xforms[i], arg.nresl - 1, wrpx)
      cnt, *lbub = evaluator(xforms[i], arg.nresl - 1, wnct)
      fn = arg.output_prefix + "_%02i.pdb" % iout
      print(
         f"{fn} score {scores[i]:7.3f} rpx {scr[0]:7.3f} cnt {cnt[0]:4}",
         f"resi {lbub[0][0]}-{lbub[1][0]}",
      )
      rp.dump_pdb_from_bodies(fn, bodies, spec.symframes(), resbounds=[lbub])
   return t.total
