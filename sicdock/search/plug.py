import threading
from time import perf_counter
from concurrent.futures import as_completed
import numpy as np
import homog as hm
from sicdock.body import Body
from sicdock.util import Bunch
from sicdock.sampling import XformHier_f4
from sicdock.io.io_body import dump_pdb_from_bodies
from sicdock.geom import xform_dist2_split
from sicdock.sym import symframes
from sicdock.bvh import bvh_isect_vec, isect_range
from sicdock.cluster import cookie_cutter

xeye = np.eye(4, dtype="f4")


def make_plugs(plug, hole, hscore, exe=None, **kw):
    opt = Bunch(kw)
    sampler = plug_get_sample_hierarchy(plug, hole, hscore)
    if opt.TEST:
        sampler = ____PLUG_TEST_SAMPLE_HIERARCHY____(plug, hole, hscore)
    plugeval = PlugEvaluator(plug, hole, hscore, **opt)
    nresl = len(hscore.hier) if opt.nresl is None else opt.nresl
    ntot, ttot, ieval = 0, perf_counter(), list()
    for iresl in range(nresl):
        indices, xforms = expand_samples(**vars())
        scores, *resbound, t = evaluate_samples(**vars())
        ieval.append((t, len(scores)))
        ntot += len(xforms)
        print(f"iresl {iresl} ntot {len(scores):11,} nonzero {np.sum(scores > 0):5,}")
    ibest = filter_redundancy(**vars())
    tdump = dump_results(**vars())
    ttot = perf_counter() - ttot
    print(f"samprate: {int(ntot / ttot):,}/s perf ttot {ttot:7.3f} tdump {tdump:7.3f}")
    print("eval time per stage:", " ".join([f"{t:8.2f}s" for t, n in ieval]))
    print("eval rate per stage:  ", " ".join([f"{int(n/t):7,}/s" for t, n in ieval]))


def evaluate_samples(exe, **kw):
    if exe:
        return evaluate_samples_exe(exe, **kw)
    else:
        return evaluate_samples_threads(**kw)


def evaluate_samples_exe(exe, plugeval, iresl, xforms, opt, **kw):
    t = perf_counter()
    assert opt.nworker > 0
    plugeval.iresl = iresl
    ntasks = int(len(xforms) / 10000)
    ntasks = min(128 * opt.nworker, max(opt.nworker, ntasks))
    futures = list()
    for i, x in enumerate(np.array_split(xforms, ntasks)):
        futures.append(exe.submit(plugeval, x))
        futures[-1].idx = i
    futures = [f for f in as_completed(futures)]
    results = [f.result() for f in sorted(futures, key=lambda x: x.idx)]
    s = np.concatenate([r[0] for r in results])
    scores = np.minimum(opt.wts.plug * s[:, 0], opt.wts.hole * s[:, 1])
    lb = np.concatenate([r[1] for r in results])
    ub = np.concatenate([r[2] for r in results])
    return scores, lb, ub, perf_counter() - t


def evaluate_samples_threads(plugeval, iresl, xforms, opt, **kw):
    t = perf_counter()
    assert opt.nworker > 0
    plugeval.iresl = iresl
    workers = [PlugWorker(xforms, plugeval, opt.nworker, i) for i in range(opt.nworker)]
    [wkr.start() for wkr in workers]
    [wkr.join() for wkr in workers]
    scores = np.empty(len(xforms))
    lb, ub = np.empty((2, len(xforms)), dtype="i4")
    for i, wkr in enumerate(workers):
        s, lb[i :: opt.nworker], ub[i :: opt.nworker] = wkr.rslt
        scores[i :: opt.nworker] = np.minimum(
            opt.wts.plug * s[:, 0], opt.wts.hole * s[:, 1]
        )
    return scores, lb, ub, perf_counter() - t


def expand_samples(iresl, sampler, opt, indices=None, scores=None, **kw):
    if iresl == 0:
        indices = np.arange(sampler.size(0), dtype="u8")
        mask, xforms = sampler.get_xforms(0, indices)
        indices = indices[mask]
    else:
        nexpand = max(1, int(opt.beam_size / 64))
        indices, xforms = sampler.expand_top_N(nexpand, iresl - 1, scores, indices)
    return indices, xforms


class PlugEvaluator:
    def __init__(self, plug, hole, hscore, wts, **kw):
        self.plug = plug.copy()
        self.plugsym = plug.copy()
        self.hole = hole
        self.hscore = hscore
        self.symrot = hm.hrot([0, 0, 1], 360 / int(hole.sym[1:]), degrees=True)
        self.iresl = 0
        self.wts = wts
        self.opt = Bunch(kw)

    def __call__(self, xpos, wts=None):
        wts = self.wts if wts is None else wts
        xpos = xpos.reshape(-1, 4, 4)
        plug, hole, iresl, hscore = self.plug, self.hole, self.iresl, self.hscore
        mindis, maxtrim, score = self.opt.clashdis, self.opt.max_trim, hscore.scorepos
        xsym = self.symrot @ xpos

        ok = np.abs((xpos @ plug.pcavecs[0])[:, 2]) <= 0.5
        ok[ok] &= ~bvh_isect_vec(plug.bvh_bb, plug.bvh_bb, xpos[ok], xsym[ok], 3.5)
        if maxtrim > 0:
            ptrim = isect_range(
                plug.bvh_bb, hole.bvh_bb, xpos[ok], xeye, mindis, maxtrim
            )
            ptrim = ((ptrim[0] - 1) // 5 + 1, (ptrim[1] + 1) // 5 - 1)  # to ptrim numb
            ntrim = ptrim[0] + plug.nres - ptrim[1] - 1
            trimok = ntrim <= maxtrim
            ptrim = (ptrim[0][trimok], ptrim[1][trimok])
            ok[ok] &= trimok
            # if np.sum(trimok) - np.sum(ntrim == 0):
            # print("ntrim not0", np.sum(trimok) - np.sum(ntrim == 0))
        else:
            phisect = bvh_isect_vec(plug.bvh_bb, hole.bvh_bb, xpos[ok], xeye, mindis)
            ok[ok] &= ~phisect
            ptrim = [0], [plug.nres - 1]

        xok = xpos[ok]
        scores = np.zeros((len(xpos), 2))
        scores[ok, 0] = score(iresl, plug, plug, xok, xsym[ok], wts, (*ptrim, *ptrim))
        scores[ok, 1] = score(iresl, plug, hole, xok, xeye[:,], wts, ptrim)
        plb = np.zeros(len(scores), dtype="i4")
        pub = np.ones(len(scores), dtype="i4") * (plug.nres - 1)
        if ptrim:
            plb[ok], pub[ok] = ptrim[0], ptrim[1]
        return scores, plb, pub


def filter_redundancy(opt, xforms, plug, scores, **kw):
    nclust = opt.max_cluster
    if nclust is None:
        nclust = int(opt.beam_size) // 10
    ibest = np.argsort(-scores)
    crd = xforms[ibest[:nclust], None] @ plug.cen[::10, :, None]
    ncen = crd.shape[1]
    crd = crd.reshape(-1, 4 * ncen)
    keep = cookie_cutter(crd, opt.rmscut * np.sqrt(ncen))
    print(f"redundancy filter cut {opt.rmscut} keep {len(keep)} of {opt.max_cluster}")
    return ibest[keep]


def dump_results(opt, xforms, plug, scores, ibest, plugeval, hole, iresl, **kw):
    t = perf_counter()
    fname_prefix = "plug" if opt.out_prefix is None else opt.out_prefix
    nout = min(10 if opt.nout is None else opt.nout, len(ibest))
    for i in range(nout):
        plug.move_to(xforms[ibest[i]])
        ((pscr, hscr),), *lbub = plugeval(xforms[ibest[i]], opt.wts.with_(ncontact=0))
        fn = fname_prefix + "_%02i.pdb" % i
        print(
            f"{fn} score {scores[ibest[i]]:7.3f} olig: {pscr:7.3f} hole: {hscr:7.3f}",
            f"resi {lbub[0][0]}-{lbub[1][0]}",
        )
        dump_pdb_from_bodies(fn, [plug], symframes(hole.sym), resbounds=[lbub])
        # dump_pdb_from_bodies(fn + "whole.pdb", [plug], symframes(hole.sym))
    dump_pdb_from_bodies("test_hole.pdb", [hole], symframes(hole.sym))
    return perf_counter() - t


class PlugWorker(threading.Thread):
    def __init__(self, xforms, plugeval, nworker, iworker):
        super().__init__(None, None, None)
        self.xforms = xforms
        self.plugeval = plugeval
        self.nworker = nworker
        self.iworker = iworker

    def run(self):
        work = range(self.iworker, len(self.xforms), self.nworker)
        self.rslt = self.plugeval(self.xforms[work])


def plug_get_sample_hierarchy(plug, hole, hscore):
    cart_samp_resl, ori_samp_resl = hscore.base.attr.xhresl
    r0 = max(hole.rg_xy(), 2 * plug.radius_max())
    nr1 = np.ceil(r0 / cart_samp_resl)
    r1 = nr1 * cart_samp_resl
    nr2 = np.ceil(r0 / cart_samp_resl * 2)
    r2 = nr2 * cart_samp_resl / 2
    nh = np.ceil(3 * hole.rg_z() / cart_samp_resl)
    h = nh * cart_samp_resl / 2
    cartub = np.array([+r2, +r2, +h])
    cartlb = np.array([-r2, -r2, -h])
    cartbs = np.array([nr2, nr2, nh], dtype="i")
    xh = XformHier_f4(cartlb, cartub, cartbs, ori_samp_resl)
    assert xh.sanity_check(), "bad xform hierarchy"
    print(f"XformHier {xh.size(0):,}", xh.cart_bs, xh.ori_resl, xh.cart_lb, xh.cart_ub)
    return xh


def ____PLUG_TEST_SAMPLE_HIERARCHY____(plug, hole, hscore):
    r, rori = hscore.base.attr.xhresl
    cartub = np.array([6 * r, r, r])
    cartlb = np.array([-6 * r, 0, 0])
    cartbs = np.array([12, 1, 1], dtype="i")
    xh = XformHier_f4(cartlb, cartub, cartbs, rori)
    assert xh.sanity_check(), "bad xform hierarchy"
    print(f"XformHier {xh.size(0):,}", xh.cart_bs, xh.ori_resl, xh.cart_lb, xh.cart_ub)
    return xh
