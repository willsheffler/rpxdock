import logging, numpy as np, rpxdock as rp, rpxdock.homog as hm
from rpxdock.search import hier_search
from rpxdock.filter import filters
from willutil import Timer, Bunch
#from icecream import ic
#ic.configureOutput(includeContext=True)

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
   logging.debug("entering make_asym()")

   kw = Bunch(kw, _strict=False)
   kw.nresl = hscore.actual_nresl if kw.nresl is None else kw.nresl
   kw.output_prefix = kw.output_prefix if kw.output_prefix else sym
   t = Timer().start()
   assert sampler is not None, 'sampler is required'

   evaluator = AsymEvaluator(bodies, hscore, **kw)
   xforms, scores, extra, stats = search(sampler, evaluator, **kw)
   ibest = rp.filter_redundancy(xforms, bodies[1], scores, **kw)

   #Not sure how to test this so leaving it commented out
   #if kw.filter_config:
   #   # Apply filters
   #   logging.debug("Applying filters to search results")
   #   sbest, filter_extra = filters.filter(xforms[ibest], bodies, **kw)
   #   ibest = ibest[sbest]

   if kw.verbose:
      print(f"rate: {int(stats.ntot / t.total):,}/s ttot {t.total:7.3f} tdump {tdump:7.3f}")
      print("stage time:", " ".join([f"{t:8.2f}s" for t, n in stats.neval]))
      print("stage rate:  ", " ".join([f"{int(n/t):7,}/s" for t, n in stats.neval]))

   xforms = xforms[ibest]
   wrpx = kw.wts.sub(rpx=1, ncontact=0)
   wnct = kw.wts.sub(rpx=0, ncontact=1)
   rpx, extra = evaluator(xforms, kw.nresl - 1, wrpx)
   ncontact, _ = evaluator(xforms, kw.nresl - 1, wnct)
   return rp.Result(
      bodies=None if kw.dont_store_body_in_results else bodies,
      attrs=dict(arg=kw, stats=stats, ttotal=t.total, sym='c1'),
      scores=(["model"], scores[ibest].astype("f4")),
      xforms=(["model", "hrow", "hcol"], xforms),
      rpx=(["model"], rpx.astype("f4")),
      ncontact=(["model"], ncontact.astype("f4")),
      reslb=(["model"], extra.reslb),
      resub=(["model"], extra.resub),
   )

class AsymEvaluator:
   def __init__(self, bodies, hscore, **kw):
      self.kw = Bunch(kw, _strict=False)
      self.bodies = bodies
      self.hscore = hscore

   def __call__(self, xforms, iresl=-1, wts={}, **kw):
      kw = self.kw.sub(wts=wts)
      xeye = np.eye(4, dtype="f4")
      body1, body2 = self.bodies
      xforms = xforms.reshape(-1, 4, 4)

      # check clash, or get non-clash range
      if kw.max_trim > 0:
         trim = body2.intersect_range(body1, xeye, xforms, **kw)
         trim, trimok = rp.search.trim_ok(trim, body2.nres, **kw)
         ok = trimok
      else:
         ok = body1.clash_ok(body2, xforms, xeye, **kw)
         trim1 = [0], [body1.nres - 1]
         trim2 = [0], [body2.nres - 1]

      # score everything that didn't clash
      scores = np.zeros(len(xforms))
      bounds = (*trim1, -1, *trim2, -1)
      scores[ok] = self.hscore.scorepos(body1, body2, xforms[ok], xeye, iresl, bounds, **kw)

      # record ranges used
      lb = np.zeros(len(scores), dtype="i4")
      # ub = np.ones(len(scores), dtype="i4") * (body1.nres - 1)
      ub = np.ones(len(scores), dtype="i4") * (body1.nres - 1)
      if kw.max_trim > 0:
         lb[ok], ub[ok] = trim2[0], trim2[1]

      return scores, Bunch(reslb=lb, resub=ub)