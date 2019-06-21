import numpy as np, xarray as xr, rpxdock as rp, rpxdock.homog as hm
from rpxdock.search import hier_search, trim_atom_to_res_numbering

def make_cyclic_hier_sampler(monomer, hscore):
   cart_resl, ori_resl = hscore.base.attr.xhresl
   ncart = int(np.ceil(2 * monomer.radius_max() / cart_resl))
   return rp.sampling.OriCart1Hier_f4([0.0], [ncart * cart_resl], [ncart], ori_resl)

_default_samplers = {hier_search: make_cyclic_hier_sampler}

def make_cyclic(monomer, sym, hscore, search=hier_search, sampler=None, **kw):
   arg = rp.Bunch(kw)
   t = rp.Timer().start()
   sym = "C%i" % i if isinstance(sym, int) else sym
   arg.nresl = len(hscore.hier) if arg.nresl is None else arg.nresl
   arg.output_prefix = arg.output_prefix if arg.output_prefix else sym

   if sampler is None: sampler = _default_samplers[search](monomer, hscore)
   evaluator = CyclicEvaluator(monomer, sym, hscore, **arg)
   xforms, scores, stats = search(sampler, evaluator, **arg)
   ibest = rp.filter_redundancy(xforms, monomer, scores, **arg)
   tdump = dump_cyclic(xforms, monomer, sym, scores, ibest, evaluator, **arg)

   if arg.verbose:
      print(f"rate: {int(stats.ntot / t.total):,}/s ttot {t.total:7.3f} tdump {tdump:7.3f}")
      print("stage time:", " ".join([f"{t:8.2f}s" for t, n in stats.neval]))
      print("stage rate:  ", " ".join([f"{int(n/t):7,}/s" for t, n in stats.neval]))

   xforms = xforms[ibest]
   wrpx = arg.wts.sub(rpx=1, ncontact=0)
   wnct = arg.wts.sub(rpx=0, ncontact=1)
   rpx, lb, ub = evaluator(xforms, arg.nresl - 1, wrpx)
   ncontact, *_ = evaluator(xforms, arg.nresl - 1, wnct)
   return rp.Result(
      body_=None if arg.dont_store_body_in_results else [monomer],
      attrs=dict(arg=arg, stats=stats, ttotal=t.total, tdump=tdump),
      scores=(["model"], scores[ibest].astype("f4")),
      xforms=(["model", "hrow", "hcol"], xforms),
      rpx=(["model"], rpx.astype("f4")),
      ncontact=(["model"], ncontact.astype("f4")),
      reslb=(["model"], lb),
      resub=(["model"], ub),
   )

class CyclicEvaluator:
   def __init__(self, body, sym, hscore, **kw):
      self.arg = rp.Bunch(kw)
      self.body = body
      self.hscore = hscore
      self.symrot = hm.hrot([0, 0, 1], 360 / int(sym[1:]), degrees=True)

   def __call__(self, xforms, iresl=-1, wts={}, **kw):
      wts = self.arg.wts.sub(wts)
      xeye = np.eye(4, dtype="f4")
      xforms = xforms.reshape(-1, 4, 4)
      body, sfxn = self.body, self.hscore.scorepos
      dclsh, max_trim = self.arg.clashdis, self.arg.max_trim
      xsym = self.symrot @ xforms

      # check for "flatness"
      ok = np.abs((xforms @ body.pcavecs[0])[:, 2]) <= self.arg.max_longaxis_dot_z

      # check clash, or get non-clash range
      if max_trim > 0:
         ptrim = body.intersect_range(body, dclsh, max_trim, xforms[ok], xsym[ok])
         ptrim, trimok = trim_atom_to_res_numbering(ptrim, body.nres, max_trim)
         ok[ok] &= trimok
      else:
         ok[ok] &= body.clash_ok(body, dclsh, xforms[ok], xsym[ok])
         ptrim = [0], [body.nres - 1]

      # score everything that didn't clash
      xok = xforms[ok]
      scores = np.zeros(len(xforms))
      scores[ok] = sfxn(body, body, xok, xsym[ok], wts, iresl, (*ptrim, *ptrim))

      # record ranges used
      plb = np.zeros(len(scores), dtype="i4")
      pub = np.ones(len(scores), dtype="i4") * (body.nres - 1)
      if ptrim:
         plb[ok], pub[ok] = ptrim[0], ptrim[1]

      return scores, plb, pub

def dump_cyclic(xforms, body, sym, scores, ibest, evaluator, **kw):
   arg = rp.Bunch(kw)
   t = rp.Timer().start()
   nout_debug = min(10 if arg.nout_debug is None else arg.nout_debug, len(ibest))
   for iout in range(nout_debug):
      i = ibest[iout]
      body.move_to(xforms[i])
      wrpx, wnct = (arg.wts.sub(rpx=1, ncontact=0), arg.wts.sub(rpx=0, ncontact=1))
      scr, *lbub = evaluator(xforms[i], arg.nresl - 1, wrpx)
      cnt, *lbub = evaluator(xforms[i], arg.nresl - 1, wnct)
      fn = arg.output_prefix + "_%02i.pdb" % iout
      print(
         f"{fn} score {scores[i]:7.3f} rpx {scr[0]:7.3f} cnt {cnt[0]:4}",
         f"resi {lbub[0][0]}-{lbub[1][0]}",
      )
      rp.dump_pdb_from_bodies(fn, [body], rp.symframes(sym), resbounds=[lbub])
   return t.total
