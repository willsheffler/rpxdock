import numpy as np, xarray as xr, rpxdock as rp, rpxdock.homog as hm
from rpxdock.search import hier_search, grid_search

def make_cyclic_hier_sampler(monomer, hscore, **kw):
   '''
   :param monomer:
   :param hscore:
   :return: 6 DOF - 2: Sampling all 3D space + moving in and out from the origin
   getting resolutions from hscore
   OriCart1Hier_f4: 3D orientations + 1D cartesion direction, Hierarchical sampling grid (4x4), where f4 is float point
   [0,0]: cartesion lb
   [ncart * cart_resl]: cartesian ub (n cartesian cells * cartesian cell width)
   [ncart]: n top level cells for sampling
   ori_resl: orientation resolution for sampling
   returns "arrays of pos" to check for a given search resolution where pos are represented by matrices
   '''
   cart_resl, ori_resl = hscore.base.attr.xhresl
   ncart = int(np.ceil(2 * monomer.radius_max() / cart_resl))
   return rp.sampling.OriCart1Hier_f4([0.0], [ncart * cart_resl], [ncart], ori_resl)

def make_cyclic_grid_sampler(monomer, cart_resl, ori_resl, **kw):
   ncart = int(np.ceil(2 * monomer.radius_max() / cart_resl))
   hiersampler = rp.sampling.OriCart1Hier_f4([0.0], [ncart * cart_resl], [ncart], ori_resl)
   isvalid, xforms = hiersampler.get_xforms(0, np.arange(hiersampler.size(0)))
   return xforms[isvalid]

_default_samplers = {hier_search: make_cyclic_hier_sampler, grid_search: make_cyclic_grid_sampler}

def make_cyclic(monomer, sym, hscore, search=None, sampler=None, **kw):
   '''
   monomer and sym are the input single unit and symmetry
   hscore (hierarchical score) defines the score functions
   Contains scores for motifs at coarse --> fine levels of search resolution
   sampler enumerates positions
   search is usually hier_search but grid_search is also available
   '''
   kw = rp.Bunch(kw)
   t = rp.Timer().start()
   sym = "C%i" % i if isinstance(sym, int) else sym
   kw.nresl = hscore.actual_nresl if kw.nresl is None else kw.nresl
   kw.output_prefix = kw.output_prefix if kw.output_prefix else sym
   if search is None:
      if kw.docking_method not in 'hier grid'.split():
         raise ValueError(f'--docking_method must be either "hier" or "grid"')
      if kw.docking_method == 'hier':
         search = hier_search
      elif kw.docking_method == 'grid':
         search = grid_search
   if sampler is None: sampler = _default_samplers[search](monomer, hscore=hscore, **kw)
   evaluator = CyclicEvaluator(monomer, sym, hscore, **kw)
   xforms, scores, extra, stats = search(sampler, evaluator, **kw)
   ibest = rp.filter_redundancy(xforms, monomer, scores, **kw)
   tdump = _debug_dump_cyclic(xforms, monomer, sym, scores, ibest, evaluator, **kw)

   if kw.verbose:
      print(f"rate: {int(stats.ntot / t.total):,}/s ttot {t.total:7.3f} tdump {tdump:7.3f}")
      print("stage time:", " ".join([f"{t:8.2f}s" for t, n in stats.neval]))
      print("stage rate:  ", " ".join([f"{int(n/t):7,}/s" for t, n in stats.neval]))

   xforms = xforms[ibest]
   '''
   dump pickle: (multidimensional pandas df) 
   body_: list of bodies/pos used in docking  
   attrs: xarray of all global config args, timing stats, total time, time to dump, and sym
   scores: weighted combined score by modelid
   xforms: xforms pos by modelid 
   rpx: rpxscore
   ncontact: ncontact score
   reslb/ub: lowerbound/upperbound of trimming
   '''
   wrpx = kw.wts.sub(rpx=1, ncontact=0)
   wnct = kw.wts.sub(rpx=0, ncontact=1)
   rpx, extra = evaluator(xforms, kw.nresl - 1, wrpx)
   ncontact, _ = evaluator(xforms, kw.nresl - 1, wnct)

   #

   print('######################################################')
   print(np.quantile(scores, [0, 0.25, 0.5, 0.75, 1.0]))
   print(np.sum(scores > 1.0), np.max(scores))
   print('######################################################')

   #

   return rp.Result(
      body_=None if kw.dont_store_body_in_results else [monomer],
      attrs=dict(arg=kw, stats=stats, ttotal=t.total, tdump=tdump, sym=sym),
      scores=(["model"], scores[ibest].astype("f4")),
      xforms=(["model", "hrow", "hcol"], xforms),
      rpx=(["model"], rpx.astype("f4")),
      ncontact=(["model"], ncontact.astype("f4")),
      reslb=(["model"], extra.reslb),
      resub=(["model"], extra.resub),
   )

class CyclicEvaluator:
   '''
   Takes a monomer position, generates a sym neighbor, and checks for "flat"-ish surface between the sym neighbors
   For trimming: does trimming thing and finds intersection until overlap isn't too overlappy/clashy anymore
   xforms: body.pos
   xsym: xforms of symmetrically related copy
   those two things get checked for intersections and clashes and scored by scorepos
   '''
   def __init__(self, body, sym, hscore, **kw):
      self.kw = rp.Bunch(kw)
      self.body = body
      self.hscore = hscore
      self.symrot = hm.hrot([0, 0, 1], 360 / int(sym[1:]), degrees=True)

   # __call__ gets called if class if called like a fcn
   def __call__(self, xforms, iresl=-1, wts={}, **kw):
      kw = self.kw.sub(wts=wts)
      xeye = np.eye(4, dtype="f4")
      body, sfxn = self.body, self.hscore.scorepos
      xforms = xforms.reshape(-1, 4, 4)  # body.pos
      xsym = self.symrot @ xforms  # symmetrized version of xforms

      # check for "flatness"
      ok = np.abs((xforms @ body.pcavecs[0])[:, 2]) <= self.kw.max_longaxis_dot_z

      # check clash, or get non-clash range
      if kw.max_trim > 0:
         trim = body.intersect_range(body, xforms[ok], xsym[ok], **kw)
         trim, trimok = rp.search.trim_ok(trim, body.nres, **kw)
         ok[ok] &= trimok
      else:
         ok[ok] &= body.clash_ok(body, xforms[ok], xsym[ok], **kw)
         trim = [0], [body.nres - 1]

      # score everything that didn't clash
      scores = np.zeros(len(xforms))
      bounds = (*trim, -1, *trim, -1)
      '''
      bounds: valid residue ranges to score after trimming i.e. don't score resi that were trimmed 
      sfxn: hscore.scorepos scores stuff from the hscore that got passed 
         takes two pos of bodies (the same monomer in this case)
         xforms: not clashing xforms 
         iresl: stage of hierarchical search (grid spacing: 4A --> 2A --> 1A --> 0.5A --> 0.25A)
         sampling at highest resl probably 0.6A due to ori + cart
         returns score # for each "dock"
      '''
      scores[ok] = sfxn(body, body, xforms[ok], xsym[ok], iresl, bounds, **kw)

      # record ranges used (trim data to return)
      lb = np.zeros(len(scores), dtype="i4")
      ub = np.ones(len(scores), dtype="i4") * (body.nres - 1)
      if trim: lb[ok], ub[ok] = trim[0], trim[1]

      # if iresl is 4:
      # sel = (scores > 136.3) * (scores < 136.306)
      # if np.any(sel): print(xforms[sel])
      # assert 0

      return scores, rp.Bunch(reslb=lb, resub=ub)

def _debug_dump_cyclic(xforms, body, sym, scores, ibest, evaluator, **kw):
   kw = rp.Bunch(kw)
   t = rp.Timer().start()
   nout_debug = min(10 if kw.nout_debug is None else kw.nout_debug, len(ibest))
   for iout in range(nout_debug):
      i = ibest[iout]
      body.move_to(xforms[i])
      wrpx, wnct = (kw.wts.sub(rpx=1, ncontact=0), kw.wts.sub(rpx=0, ncontact=1))
      scr, extra = evaluator(xforms[i], kw.nresl - 1, wrpx)
      cnt, extra = evaluator(xforms[i], kw.nresl - 1, wnct)
      fn = kw.output_prefix + "_%02i.pdb" % iout
      print(
         f"{fn} score {scores[i]:7.3f} rpx {scr[0]:7.3f} cnt {cnt[0]:4}",
         f"resi {extra.reslb[0]}-{extra.resub[0]}",
      )
      rp.dump_pdb_from_bodies(fn, [body], rp.symframes(sym), resbounds=extra)
   return t.total
