import numpy as np
import rpxdock as rp
import rpxdock.homog as hm
from rpxdock.search import hier_search, grid_search
from rpxdock.filter import filters
import willutil as wu

def make_cyclic_stack_hier_sampler(sym, monomer, hscore, **kw):
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
    kw = wu.Bunch(kw)
    nfold = float(sym[1:])
    # cart_resl, ori_resl = hscore.base.attr.xhresl
    if kw.limit_rotation_to_z:
        raise NotImplementedError
        maxcart = monomer.radius_max() * nfold / 3
        cycsamp = rp.sampling.RotCart1Hier_f4(0.0,
                                              maxcart,
                                              int(maxcart),
                                              0.0,
                                              360.0,
                                              360,
                                              axis=[0, 0, 1],
                                              cartaxis=[1, 0, 0])
    elif kw.disable_rotation:
        raise NotImplementedError
        maxcart = monomer.radius_max() * nfold / 3
        cycsamp = rp.sampling.CartHier2D_f4([-maxcart, maxcart], [-maxcart, maxcart], int(maxcart / 4))
    else:
        lb, ub = 0, nfold / 3 * monomer.radius_max()
        if len(kw.cart_bounds) == 3: lb, ub = kw.cart_bounds[0]
        ncart = max(1, int(np.ceil((ub - lb) / kw.cart_resl)))
        cycsamp = rp.sampling.OriCart1Hier_f4([lb], [ub], [ncart], kw.ori_resl)

    # indices = np.arange(cycsamp.size(0), dtype='u8')
    # mask, xforms = cycsamp.get_xforms(0, indices)
    # wu.showme(xforms)

    ang_range = 360 / nfold
    lb, ub = 0, 3 * monomer.radius_max()
    if len(kw.cart_bounds) == 3: lb, ub = kw.cart_bounds[2]
    stacksamp = rp.sampling.RotCart1Hier_f4(lb, ub, max(1, int((ub - lb) / kw.cart_resl)), 0.0, ang_range,
                                            int(ang_range / kw.ori_resl))
    # stacksamp = rp.sampling.ZeroDHier(wu.htrans([0, 0, 1]))

    return rp.sampling.CompoundHier(cycsamp, stacksamp)

def make_cyclic_stack(monomer, sym, hscore, **kw):
    kw = wu.Bunch(kw, _strict=False)
    t = wu.Timer().start()
    sym = "C%i" % sym if isinstance(sym, int) else sym
    kw.nresl = hscore.actual_nresl if kw.nresl is None else kw.nresl
    kw.output_prefix = kw.output_prefix if kw.output_prefix else sym
    sampler = make_cyclic_stack_hier_sampler(sym, monomer, hscore, **kw)
    evaluator = CyclicStackEvaluator(monomer, sym, hscore, **kw)

    if 0:
        x = sampler.get_xforms(0, np.arange(sampler.size(0)))[1]
        ic(x.shape)
        wu.showme(x[:, 0] @ wu.htrans([1, 2, 3]))
        wu.showme(evaluator.symrot @ x[:, 0] @ wu.htrans([1, 2, 3]))
        # wu.showme(x[:, 1] @ x[:, 0] @ wu.htrans([1, 2, 3]))
        assert 0

    xforms, scores, extra, stats = hier_search(sampler, evaluator, **kw)
    ibest = rp.filter_redundancy(xforms, [monomer] * 2, scores, symframes=sym, **kw)
    tdump = _debug_dump_cyclic_stack(xforms, [monomer] * 2, sym, scores, ibest, evaluator, **kw)

    if kw.verbose:
        print(f"rate: {int(stats.ntot / t.total):,}/s ttot {t.total:7.3f} tdump {tdump:7.3f}")
        print("stage time:", " ".join([f"{t:8.2f}s" for t, n in stats.neval]))
        print("stage rate:  ", " ".join([f"{int(n/t):7,}/s" for t, n in stats.neval]))

    if kw.filter_config:
        # Apply filters
        sbest, filter_extra = filters.filter(xforms[ibest], monomer, **kw)
        ibest = ibest[sbest]

    xforms = xforms[ibest]
    xforms[:, 1] = xforms[:, 1] @ xforms[:, 0]
    wrpx = kw.wts.sub(rpx=1, ncontact=0)
    wnct = kw.wts.sub(rpx=0, ncontact=1)
    rpx, extra = evaluator(xforms, kw.nresl - 1, wrpx)
    ncontact, _ = evaluator(xforms, kw.nresl - 1, wnct)

    data = dict(
        attrs=dict(arg=kw, stats=stats, ttotal=t.total, tdump=tdump, sym=sym),
        scores=(["model"], scores[ibest].astype("f4")),
        xforms=(["model", "oristack", "hrow", "hcol"], xforms),
        rpx=(["model"], rpx.astype("f4")),
        ncontact=(["model"], ncontact.astype("f4")),
    )

    for k, v in extra.items():
        if not isinstance(v, (list, tuple)) or len(v) > 3:
            v = ['model'], v
        data[k] = v

    if kw.filter_config:
        #add the filter data to data
        for k, v in filter_extra.items():
            if not isinstance(v, (list, tuple)) or len(v) > 3:
                v = ['model'], v
            data[k] = v

    return rp.Result(
        bodies=None if kw.dont_store_body_in_results else [[monomer, monomer.copy()]],
        **data,
    )

class CyclicStackEvaluator:
    '''
   Takes a monomer position, generates a sym neighbor, and checks for "flat"-ish surface between the sym neighbors
   For trimming: does trimming thing and finds intersection until overlap isn't too overlappy/clashy anymore
   xforms: body.pos
   xsym: xforms of symmetrically related copy
   those two things get checked for intersections and clashes and scored by scorepos
   '''
    def __init__(self, body, sym, hscore, **kw):
        self.kw = wu.Bunch(kw, _strict=False)
        self.body = body
        self.hscore = hscore
        self.symrot = hm.hrot([0, 0, 1], 360 / int(sym[1:]), degrees=True)

    # __call__ gets called if class if called like a fcn
    def __call__(self, xforms, iresl=-1, wts={}, **kw):
        kw = self.kw.sub(wts=wts)
        # xeye = np.eye(4, dtype="f4")
        body, sfxn = self.body, self.hscore.scorepos
        xori = xforms[:, 0]
        xsym = self.symrot @ xori  # symmetrized version of xforms
        xstack = xforms[:, 1] @ xori
        kw.mindis = kw.clash_distances[iresl]

        # check for "flatness"
        ok = np.abs((xori @ body.pcavecs[0])[:, 2]) <= self.kw.max_longaxis_dot_z
        ok[ok] &= body.clash_ok(body, xori[ok], xsym[ok], **kw)
        ok[ok] &= body.clash_ok(body, xori[ok], xstack[ok], **kw)
        ok[ok] &= body.clash_ok(body, xsym[ok], xstack[ok], **kw)

        # score everything that didn't clash
        scores = np.zeros(len(xori))
        scores[ok] += sfxn(body, body, xori[ok], xsym[ok], iresl, **kw)
        scores[ok] += sfxn(body, body, xori[ok], xstack[ok], iresl, **kw)
        scores[ok] += sfxn(body, body, xsym[ok], xstack[ok], iresl, **kw)

        return scores, wu.Bunch()

def _debug_dump_cyclic_stack(xforms, body, sym, scores, ibest, evaluator, **kw):
    kw = wu.Bunch(kw, _strict=False)
    t = wu.Timer().start()
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
