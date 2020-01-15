import logging, numpy as np, xarray as xr, rpxdock as rp, rpxdock.homog as hm
from rpxdock.search import hier_search

log = logging.getLogger(__name__)

def asym_get_sample_hierarchy(body, hscore, extent=100):
   "set up XformHier with appropriate bounds and resolution"
   cart_xhresl, ori_xhresl = hscore.base.attr.xhresl
   rg = body.rg()
   cart_samp_resl = 0.707 * cart_xhresl
   ori_samp_resl = cart_samp_resl / rg * 180 / np.pi
   # print(cart_samp_resl, rg, ori_samp_resl)
   ori_samp_resl = min(ori_samp_resl, ori_xhresl)
   # print(f"asym_get_sample_hierarchy cart: {cart_samp_resl} ori: {ori_samp_resl}")
   ncart = np.ceil(extent * 2 / cart_samp_resl)
   cartlb = np.array([-extent] * 3)
   cartub = np.array([extent] * 3)
   cartbs = np.array([ncart] * 3, dtype="i")
   xh = rp.sampling.XformHier_f4(cartlb, cartub, cartbs, ori_samp_resl)
   assert xh.sanity_check(), "bad xform hierarchy"
   log.info(f"XformHier {xh.size(0):,} {xh.cart_bs} {xh.ori_resl} {xh.cart_lb} {xh.cart_ub}")
   return xh

def make_asym(bodies, hscore, sampler, search=hier_search, **kw):
   arg = rp.Bunch(kw)
   arg.nresl = hscore.actual_nresl if arg.nresl is None else arg.nresl
   arg.output_prefix = arg.output_prefix if arg.output_prefix else sym
   t = rp.Timer().start()
   assert sampler is not None, 'sampler is required'

   evaluator = AsymEvaluator(bodies, hscore, **arg)
   xforms, scores, extra, stats = search(sampler, evaluator, **arg)
   ibest = rp.filter_redundancy(xforms, bodies[1], scores, **arg)

   if arg.verbose:
      print(f"rate: {int(stats.ntot / t.total):,}/s ttot {t.total:7.3f} tdump {tdump:7.3f}")
      print("stage time:", " ".join([f"{t:8.2f}s" for t, n in stats.neval]))
      print("stage rate:  ", " ".join([f"{int(n/t):7,}/s" for t, n in stats.neval]))

   xforms = xforms[ibest]
   wrpx = arg.wts.sub(rpx=1, ncontact=0)
   wnct = arg.wts.sub(rpx=0, ncontact=1)
   rpx, extra = evaluator(xforms, arg.nresl - 1, wrpx)
   ncontact, _ = evaluator(xforms, arg.nresl - 1, wnct)
   return rp.Result(
      body_=None if arg.dont_store_body_in_results else bodies,
      attrs=dict(arg=arg, stats=stats, ttotal=t.total),
      scores=(["model"], scores[ibest].astype("f4")),
      xforms=(["model", "hrow", "hcol"], xforms),
      rpx=(["model"], rpx.astype("f4")),
      ncontact=(["model"], ncontact.astype("f4")),
      reslb=(["model"], extra.reslb),
      resub=(["model"], extra.resub),
   )

class AsymEvaluator:
   def __init__(self, bodies, hscore, **kw):
      self.arg = rp.Bunch(kw)
      self.bodies = bodies
      self.hscore = hscore

   def __call__(self, xforms, iresl=-1, wts={}, **kw):
      arg = self.arg.sub(wts=wts)
      xeye = np.eye(4, dtype="f4")
      body1, body2 = self.bodies
      xforms = xforms.reshape(-1, 4, 4)

      # check clash, or get non-clash range
      if arg.max_trim > 0:
         trim = body2.intersect_range(body1, xeye, xforms, **arg)
         trim, trimok = rp.search.trim_ok(trim, body2.nres, **arg)
         ok = trimok
      else:
         ok = body1.clash_ok(body2, xforms, xeye, **arg)
         trim = [0], [body2.nres - 1]

      # score everything that didn't clash
      scores = np.zeros(len(xforms))
      bounds = (*trim, -1, *trim, -1)
      scores[ok] = self.hscore.scorepos(body1, body2, xforms[ok], xeye, iresl, bounds, **arg)
      # scores[ok] = self.hscore.scorepos(body1, body2, xeye, xforms[ok], iresl, bounds, **arg)

      # record ranges used
      lb = np.zeros(len(scores), dtype="i4")
      # ub = np.ones(len(scores), dtype="i4") * (body1.nres - 1)
      ub = np.ones(len(scores), dtype="i4") * (body2.nres - 1)
      if trim: lb[ok], ub[ok] = trim[0], trim[1]

      return scores, rp.Bunch(reslb=lb, resub=ub)
