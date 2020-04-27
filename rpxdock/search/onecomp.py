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
   '''
   :param body: pose info
   :param spec: architecture info
   :param hscore: motif stuff
   :param search: type of search
   :param sampler: defined as None, but we feed it hier_axis_sampler
   :param fixed_components: whether one component is fixed
   :param kw: all default variables we are bringing in
   :return:
   '''
   arg = rp.Bunch(kw)
   t = rp.Timer().start()
   arg.nresl = hscore.actual_nresl if arg.nresl is None else arg.nresl # number of steps in hier search
   arg.output_prefix = arg.output_prefix if arg.output_prefix else spec.arch

   assert isinstance(body, rp.Body)
   if not fixed_components:
      body = body.copy_xformed(rp.homog.align_vector([0, 0, 1], spec.axis)) # align body axis of symmetry to z axis

   dotrim = arg.max_trim and arg.trimmable_components # maximum of residues and which component to trim
   evaluator = OneCompEvaluator(body, spec, hscore, **arg) #initiate instance of evaluator (set up geom and score)
   xforms, scores, extra, stats = search(sampler, evaluator, **arg) #search stuff given pos and scores of evaluator and get positions, scores, and other stuff
   ibest = rp.filter_redundancy(xforms, body, scores, **arg) #orders results from best to worst and checks for redundancy by score
   # tdump = _debug_dump_cage(xforms, body, spec, scores, ibest, evaluator, **arg) # debugging, but is commented out in this case?

   # spitting out random shit. we don't really care, compute time stats and crap
   if arg.verbose:
      print(f"rate: {int(stats.ntot / t.total):,}/s ttot {t.total:7.3f}")
      print("stage time:", " ".join([f"{t:8.2f}s" for t, n in stats.neval]))
      print("stage rate:  ", " ".join([f"{int(n/t):7,}/s" for t, n in stats.neval]))

   xforms = xforms[ibest] # best transforms from the evaluator
   wrpx = arg.wts.sub(rpx=1, ncontact=0) # rpx score weights
   wnct = arg.wts.sub(rpx=0, ncontact=1) # ncontact weights
   rpx, extra = evaluator(xforms, arg.nresl - 1, wrpx) # gives actual rpxscore
   ncontact, *_ = evaluator(xforms, arg.nresl - 1, wnct) # gets actual ncontacts
   data = dict(
      attrs=dict(arg=arg, stats=stats, ttotal=t.total, output_prefix=arg.output_prefix,
                 output_body='all', sym=spec.arch),
      scores=(["model"], scores[ibest].astype("f4")),
      xforms=(["model", "hrow", "hcol"], xforms),
      rpx=(["model"], rpx.astype("f4")),
      ncontact=(["model"], ncontact.astype("f4")),
   )

   # put additional geom stuff into data
   for k, v in extra.items():
      if not isinstance(v, (list, tuple)) or len(v) > 3:
         v = ['model'], v
      data[k] = v
   data['disp'] = (['model'], np.sum(xforms[:, :3, 3] * spec.axis[None, :3], axis=-1).squeeze())
   data['angle'] = (['model'], rp.homog.angle_of(xforms[:]) * 180 / np.pi)
   default_label = ['compA']

   return rp.Result(
      body_=None if arg.dont_store_body_in_results else [body],
      body_label_=[] if arg.dont_store_body_in_results else default_label,
      **data,
   )

class OneCompEvaluator:
   '''
   Takes a monomer position, generates a sym neighbor, and checks for clashes between the sym neighbors
   For trimming: does trimming thing and finds intersection until overlap isn't too overlappy/clashy anymore
   xforms: body.pos
   xsym: xforms of symmetrically related copy
   those two things get checked for intersections and clashes and scored by scorepos
   Does not check for flatness like Cyclic, because cages aren't flat.
   '''
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

      '''
      bounds: valid residue ranges to score after trimming i.e. don't score resi that were trimmed 
      sfxn: hscore.scorepos scores stuff from the hscore that got passed 
         takes two pos of bodies (the same monomer in this case)
         xforms: not clashing xforms 
         iresl: stage of hierarchical search (grid spacing: 4A --> 2A --> 1A --> 0.5A --> 0.25A)
         sampling at highest resl probably 0.6A due to ori + cart
         returns score # for each "dock"
      '''

      # record ranges used
      lb = np.zeros(len(scores), dtype="i4")
      ub = np.ones(len(scores), dtype="i4") * (body.nres - 1)
      if trim: lb[ok], ub[ok] = trim[0], trim[1]

      return scores, rp.Bunch(reslb=lb, resub=ub)
