import numpy as np, xarray as xr, rpxdock as rp, rpxdock.homog as hm
from rpxdock.search import hier_search

def make_cyclic_hier_sampler(monomer, hscore):
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

_default_samplers = {hier_search: make_cyclic_hier_sampler}

def make_cyclic(monomer, sym, hscore, search=hier_search, sampler=None, **kw):
   '''
   monomer and sym are the input single unit and symmetry
   hscore (hierarchical score) defines the score functions
   Contains scores for motifs at coarse --> fine levels of search resolution
   sampler enumerates positions
   search is usually hier_search but grid_search is also available
   '''
   arg = rp.Bunch(kw) #options
   t = rp.Timer().start()
   sym = "C%i" % i if isinstance(sym, int) else sym
   arg.nresl = hscore.actual_nresl if arg.nresl is None else arg.nresl #number of steps in hier search
   arg.output_prefix = arg.output_prefix if arg.output_prefix else sym

   if sampler is None: sampler = _default_samplers[search](monomer, hscore) #checks list of pos for monomer
   evaluator = CyclicEvaluator(monomer, sym, hscore, **arg) #initiate instance of evaluator (set up geom and score)
   xforms, scores, extra, stats = search(sampler, evaluator, **arg) #search stuff given pos and scores of evaluator and get positions, scores, and other stuff
   ibest = rp.filter_redundancy(xforms, monomer, scores, **arg) #orders results from best to worst and checks for redundancy by score
   tdump = _debug_dump_cyclic(xforms, monomer, sym, scores, ibest, evaluator, **arg)

   if arg.verbose:
      print(f"rate: {int(stats.ntot / t.total):,}/s ttot {t.total:7.3f} tdump {tdump:7.3f}")
      print("stage time:", " ".join([f"{t:8.2f}s" for t, n in stats.neval]))
      print("stage rate:  ", " ".join([f"{int(n/t):7,}/s" for t, n in stats.neval]))

   xforms = xforms[ibest]
   wrpx = arg.wts.sub(rpx=1, ncontact=0) # compute weighted rpx
   wnct = arg.wts.sub(rpx=0, ncontact=1) # compute weighted ncontact
   rpx, extra = evaluator(xforms, arg.nresl - 1, wrpx) # reevaluate scores as components
   ncontact, _ = evaluator(xforms, arg.nresl - 1, wnct)

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
   return rp.Result(
      body_=None if arg.dont_store_body_in_results else [monomer],
      attrs=dict(arg=arg, stats=stats, ttotal=t.total, tdump=tdump, sym=sym),
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
      self.arg = rp.Bunch(kw)
      self.body = body
      self.hscore = hscore
      self.symrot = hm.hrot([0, 0, 1], 360 / int(sym[1:]), degrees=True)

   # __call__ gets called if class if called like a fcn
   def __call__(self, xforms, iresl=-1, wts={}, **kw):
      arg = self.arg.sub(wts=wts)
      xeye = np.eye(4, dtype="f4")
      body, sfxn = self.body, self.hscore.scorepos
      xforms = xforms.reshape(-1, 4, 4)  # body.pos
      xsym = self.symrot @ xforms # symmetrized version of xforms

      # check for "flatness"
      ok = np.abs((xforms @ body.pcavecs[0])[:, 2]) <= self.arg.max_longaxis_dot_z

      # check clash, or get non-clash range
      if arg.max_trim > 0:
         trim = body.intersect_range(body, xforms[ok], xsym[ok], **arg) # what residues can you have w/o clashing
         trim, trimok = rp.search.trim_ok(trim, body.nres, **arg)
         ok[ok] &= trimok # given an array of pos/xforms, filter out pos/xforms that clash
      else:
         ok[ok] &= body.clash_ok(body, xforms[ok], xsym[ok], **arg) # if no trim, just checks for clashes (intersecting)
         trim = [0], [body.nres - 1]  # no trimming

      # score everything that didn't clash
      scores = np.zeros(len(xforms))
      bounds = (*trim, -1, *trim, -1)
      scores[ok] = sfxn(body, body, xforms[ok], xsym[ok], iresl, bounds, **arg)
      '''
      bounds: valid residue ranges to score after trimming i.e. don't score resi that were trimmed 
      sfxn: hscore.scorepos scores stuff from the hscore that got passed 
         takes two pos of bodies (the same monomer in this case)
         xforms: not clashing xforms 
         iresl: stage of hierarchical search (grid spacing: 4A --> 2A --> 1A --> 0.5A --> 0.25A)
         sampling at highest resl probably 0.6A due to ori + cart
         returns score # for each "dock"
      '''

      # record ranges used (trim data to return)
      lb = np.zeros(len(scores), dtype="i4")
      ub = np.ones(len(scores), dtype="i4") * (body.nres - 1)
      if trim: lb[ok], ub[ok] = trim[0], trim[1]

      return scores, rp.Bunch(reslb=lb, resub=ub)

def _debug_dump_cyclic(xforms, body, sym, scores, ibest, evaluator, **kw):
   arg = rp.Bunch(kw)
   t = rp.Timer().start()
   nout_debug = min(10 if arg.nout_debug is None else arg.nout_debug, len(ibest))
   for iout in range(nout_debug):
      i = ibest[iout]
      body.move_to(xforms[i])
      wrpx, wnct = (arg.wts.sub(rpx=1, ncontact=0), arg.wts.sub(rpx=0, ncontact=1))
      scr, extra = evaluator(xforms[i], arg.nresl - 1, wrpx)
      cnt, extra = evaluator(xforms[i], arg.nresl - 1, wnct)
      fn = arg.output_prefix + "_%02i.pdb" % iout
      print(
         f"{fn} score {scores[i]:7.3f} rpx {scr[0]:7.3f} cnt {cnt[0]:4}",
         f"resi {extra.reslb[0]}-{extra.resub[0]}",
      )
      rp.dump_pdb_from_bodies(fn, [body], rp.symframes(sym), resbounds=extra)
   return t.total
