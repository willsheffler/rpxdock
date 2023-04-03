import logging, numpy as np, rpxdock as rp, rpxdock.homog as hm
from rpxdock.search import hier_search
from rpxdock.filter import filters
import willutil as wu
#from icecream import ic
#ic.configureOutput(includeContext=True)

log = logging.getLogger(__name__)

def isnt_used_huh_asym_get_sample_hierarchy(body, hscore, extent=100, frames=None, x2asymcen=None):
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

@wu.timed
def make_asym(bodies, hscore, sampler, search=hier_search, frames=None, x2asymcen=None, sym='C1', **kw):
   logging.debug("entering make_asym()")
   if isinstance(bodies, rp.body.Body): bodies = [bodies]

   kw = wu.Bunch(kw, _strict=False)
   kw.nresl = hscore.actual_nresl if kw.nresl is None else kw.nresl
   kw.output_prefix = kw.output_prefix if kw.output_prefix else sym
   t = wu.Timer().start()
   assert sampler is not None, 'sampler is required'

   if frames is not None:
      evaluator = AsymFramesEvaluator(bodies, hscore, frames=frames, x2asymcen=x2asymcen, **kw)
   else:
      evaluator = AsymEvaluator(bodies, hscore, **kw)
   wu.checkpoint(kw, 'make_asym')
   xforms, scores, extra, stats = search(sampler, evaluator, **kw)
   wu.checkpoint(kw, 'search')
   xforms = evaluator.modify_xforms(xforms)
   if frames is not None:
      xforms = wu.hxform(wu.hinv(x2asymcen), wu.hxform(xforms, x2asymcen))

   ibest = np.argsort(scores)
   if kw.max_bb_redundancy > 0:
      if frames is None:
         ibest = rp.filter_redundancy(xforms, bodies[1], scores, **kw)
      else:
         ibest = rp.filter_redundancy(xforms, bodies[0][0], scores, **kw)

   if frames is not None:
      if 'filter_sscount' in kw:
         # ic(bodies)
         for i, frame in enumerate(frames[1:]):
            sbest, filter_extra = rp.filter.sscount.filter_sscount(
               bodiespos=(bodies[0][0], bodies[0][0], xforms[ibest], wu.hxform(frame, xforms[ibest])), **kw)
            ibest = ibest[sbest]
            # ic(np.sum(sbest), filter_extra)

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
   # wrpx = kw.wts.sub(rpx=1, ncontact=0)
   # wnct = kw.wts.sub(rpx=0, ncontact=1)
   # if frames is None:
   # rpx, extra = evaluator(xforms, kw.nresl - 1, wrpx)
   # ncontact, _ = evaluator(xforms, kw.nresl - 1, wnct)

   resultbodies = bodies if frames is None else bodies[0][0]
   result = dict(
      bodies=None if kw.dont_store_body_in_results else resultbodies,
      attrs=dict(arg=kw, stats=stats, ttotal=t.total, sym='c1'),
      scores=(["model"], scores[ibest].astype("f4")),
      xforms=(["model", "hrow", "hcol"], xforms),
   )
   if 'reslb' in extra: result['reslb'] = (["model"], extra.reslb)
   if 'resub' in extra: result['resub'] = (["model"], extra.resub)
   return rp.Result(**result)

class AsymFramesEvaluator:
   def __init__(
         self,
         bodies,
         hscore,
         frames=[np.eye(4)],
         clashframes=None,
         x2asymcen=np.eye(4),
         limit_rotation=0,
         limit_translation=0,
         clashdist=3,
         scale_translation=None,
         **kw,
   ):
      self.kw = wu.Bunch(kw, _strict=False)
      self.bodies0, self.bodies1 = bodies
      self.hscore = hscore
      self.frames = frames
      self.x2asymcen = x2asymcen
      self.limit_rotation = limit_rotation
      self.limit_translation = limit_translation
      # self.body2 = body.copy()
      # self.body2.required_res_sets = list()
      self.clashdist = clashdist
      self.scale_translation = None
      self.clashframes = frames if clashframes is None else clashframes
      if isinstance(scale_translation, (int, float)):
         self.scale_translation = scale_translation * wu.hnormalized(wu.hcom(bodies0[0]))

   def modify_xforms(self, xforms):
      if self.scale_translation is None:
         return xforms
      xforms2 = xforms.copy()
      scalemag = wu.hnorm(self.scale_translation)
      scaledir = wu.hnormalized(self.scale_translation)
      p = wu.hproj(self.biasdir, wu.hcart3(xforms2))
      pp = wu.hprojperp(scaledir, wu.hcart3(xforms2))
      trans = p[:3] * scalemag + pp[:3]
      xforms2[:3, 3] = trans
      assert hvalid(xforms2)
      assert xforms2.shape == xforms.shape

      assert 0
      return xforms2

   def __call__(self, xforms, iresl=-1, wts={}, **kw):
      kw = self.kw.sub(wts=wts)
      xforms = xforms.reshape(-1, 4, 4)

      ok = np.ones(len(xforms), dtype=np.bool)
      if self.limit_rotation > 0:
         _, ang = wu.haxis_angle_of(xforms[ok])
         ok[ok] = ang <= self.limit_rotation + [0.0, 0.1, 0.2, 0.3, 0.5, 0.5, 0.5][iresl]
      if self.limit_translation > 0:
         ok[ok] = wu.hnorm(xforms[ok, :3, 3]) <= self.limit_translation + [0, 1, 2, 4, 8, 8, 8][iresl]
      if np.sum(ok) == 0:
         return np.zeros(len(xforms)), wu.Bunch()

      xasym = wu.hinv(self.x2asymcen) @ xforms @ self.x2asymcen

      clashdist = [5, 3, 2, 1.5, 1.5, 1.5, 1.5][iresl]
      # clashdist = [5, 3, 2, 2, 2, 2, 2][iresl]
      # clashdist = [1, 1, 1, 1, 1, 1, 1][iresl]
      # clashdist = [4, 1, 1, 1, 1, 1, 1][iresl]
      for iframe, xframe in enumerate(self.clashframes[1:]):
         ok[ok] = self.bodies0[0].clash_ok(
            self.bodies0[0],
            self.frames[0] @ xasym[ok],
            xframe @ xasym[ok],
            mindis=clashdist,
            **kw,
         )
         if np.sum(ok) == 0:
            return np.zeros(len(xforms)), wu.Bunch()
         # ic(iresl, iframe, np.sum(ok))
      # TODO: fix this into one loop...

      # why was this necessary?
      #for iframe, xframe in enumerate(self.frames[2:]):
      #   ok[ok] = self.bodies0[0].clash_ok(
      #      self.bodies0[0],
      #      self.frames[1] @ xasym[ok],
      #      xframe @ xasym[ok],
      #      mindis=self.clashdist,
      #      **kw,
      #   )
      # score everything that didn't clash
      scores = np.zeros(len(xforms))
      for ibody in range(len(self.bodies0)):
         newscores = self.hscore.scorepos(
            self.bodies0[ibody],
            self.bodies1[ibody],
            self.frames[0] @ xasym[ok],
            self.frames[ibody + 1] @ xasym[ok],
            iresl,
            ibody=ibody,
            **kw,
         )
         if ibody == 0:
            scores[ok] = newscores
         else:
            # ic(ibody)
            # if ibody == 2: newscores *= 0.5
            minsc = np.minimum(scores[ok], newscores)
            # ic(minsc.shape, scores[ok].shape)
            scores[ok] = minsc
         # scores[ok] = scores[ok] + newscores
         ok[ok] = np.logical_and(ok[ok], newscores > 0)
      if np.sum(scores > 0) == 0:
         # raise ValueError(f'no results at stage {iresl}')
         print(f'warning: no results at stape {iresl}', flush=True)
      return scores, wu.Bunch()

class AsymEvaluator:
   def __init__(self, bodies, hscore, **kw):
      self.kw = wu.Bunch(kw, _strict=False)
      self.bodies = bodies
      self.hscore = hscore

   def modify_xforms(self, xforms):
      return xforms

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
      return scores, wu.Bunch(reslb=lb, resub=ub)