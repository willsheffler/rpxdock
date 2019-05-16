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
from sicdock.bvh import bvh_isect_vec
from sicdock.cluster import cookie_cutter

xeye = np.eye(4, dtype="f4")


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


def make_plugs(plug, hole, hscore, **kw):
    args = Bunch(kw)
    sampler = plug_get_sample_hierarchy(plug, hole, hscore)
    if args.TEST:
        sampler = ____PLUG_TEST_SAMPLE_HIERARCHY____(plug, hole, hscore)
    plugeval = PlugEvaluator(plug, hole, hscore, **args)
    nresl = len(hscore.hier) if args.nresl is None else args.nresl

    ttot = perf_counter()
    xforms, scores, stats = sample_plugs(**args.sub(vars()))
    ibest = filter_redundancy(**args.sub(**vars()))
    tdump = dump_results(**args.sub(vars()))
    ttot = perf_counter() - ttot

    print(f"rate: {int(stats.ntot / ttot):,}/s ttot {ttot:7.3f} tdump {tdump:7.3f}")
    print("stage time:", " ".join([f"{t:8.2f}s" for t, n in stats.neval]))
    print("stage rate:  ", " ".join([f"{int(n/t):7,}/s" for t, n in stats.neval]))


def sample_plugs(plug, hole, hscore, sampler, nresl, plugeval, **kw):
    args = Bunch(kw)
    neval = list()
    for iresl in range(nresl):
        indices, xforms = expand_samples(**args.sub(vars()))
        scores, *resbound, t = evaluate_samples(**args.sub(vars()))
        neval.append((t, len(scores)))
        print(f"iresl {iresl} ntot {len(scores):11,} nonzero {np.sum(scores > 0):5,}")
    stats = Bunch(ntot=sum(x[1] for x in neval), neval=neval)
    return xforms, scores, stats


def evaluate_samples(**kw):
    if kw["exe"]:
        return evaluate_samples_exe(**kw)
    else:
        return evaluate_samples_threads(**kw)


def evaluate_samples_exe(exe, plugeval, iresl, xforms, nworker, wts, **kw):
    t = perf_counter()
    assert nworker > 0
    plugeval.iresl = iresl
    ntasks = int(len(xforms) / 10000)
    ntasks = min(128 * nworker, max(nworker, ntasks))
    futures = list()
    for i, x in enumerate(np.array_split(xforms, ntasks)):
        futures.append(exe.submit(plugeval, x))
        futures[-1].idx = i
    futures = [f for f in as_completed(futures)]
    results = [f.result() for f in sorted(futures, key=lambda x: x.idx)]
    s = np.concatenate([r[0] for r in results])
    scores = kw["multi_iface_summary"](s * (wts.plug, wts.hole), axis=1)
    lb = np.concatenate([r[1] for r in results])
    ub = np.concatenate([r[2] for r in results])
    return scores, lb, ub, perf_counter() - t


# def evaluate_samples_threads(plugeval, iresl, xforms, args, **kw):
#     t = perf_counter()
#     assert args.nworker > 0
#     plugeval.iresl = iresl
#     workers = [
#         PlugWorker(xforms, plugeval, args.nworker, i) for i in range(args.nworker)
#     ]
#     [wkr.start() for wkr in workers]
#     [wkr.join() for wkr in workers]
#     scores = np.empty(len(xforms))
#     lb, ub = np.empty((2, len(xforms)), dtype="i4")
#     for i, wkr in enumerate(workers):
#         s, lb[i :: args.nworker], ub[i :: args.nworker] = wkr.rslt
#         scores[i :: args.nworker] = np.minimum(
#             args.wts.plug * s[:, 0], args.wts.hole * s[:, 1]
#         )
#     return scores, lb, ub, perf_counter() - t


def expand_samples(iresl, sampler, beam_size, indices=None, scores=None, **kw):
    if iresl == 0:
        indices = np.arange(sampler.size(0), dtype="u8")
        mask, xforms = sampler.get_xforms(0, indices)
        indices = indices[mask]
    else:
        nexpand = max(1, int(beam_size / 64))
        indices, xforms = sampler.expand_top_N(nexpand, iresl - 1, scores, indices)
    return indices, xforms


class PlugEvaluator:
    def __init__(self, plug, hole, hscore, **kw):
        self.plug = plug.copy()
        self.plugsym = plug.copy()
        self.hole = hole
        self.hscore = hscore
        self.symrot = hm.hrot([0, 0, 1], 360 / int(hole.sym[1:]), degrees=True)
        self.iresl = None  # must be set in client code
        self.wts = kw["wts"]
        self.clashdis = kw["clashdis"]
        self.max_trim = kw["max_trim"]
        self.max_longaxis_dot_z = kw["max_longaxis_dot_z"]

    def __call__(self, xpos, wts={}):
        wts = self.wts.sub(wts)
        xpos = xpos.reshape(-1, 4, 4)
        plug, hole, iresl, hscore = self.plug, self.hole, self.iresl, self.hscore
        clashdis, max_trim, score = self.clashdis, self.max_trim, hscore.scorepos
        xsym = self.symrot @ xpos

        # check for "flatness"
        ok = np.abs((xpos @ plug.pcavecs[0])[:, 2]) <= self.max_longaxis_dot_z

        # check chash in formed oligomer
        ok[ok] &= plug.clash_ok(plug, clashdis, xpos[ok], xsym[ok])

        # check clash olig vs hole, or get non-clash range
        if max_trim > 0:
            ptrim = plug.intersect_range(hole, clashdis, max_trim, xpos[ok])
            ptrim = ((ptrim[0] - 1) // 5 + 1, (ptrim[1] + 1) // 5 - 1)  # to res numbers
            ntrim = ptrim[0] + plug.nres - ptrim[1] - 1
            trimok = ntrim <= max_trim
            ptrim = (ptrim[0][trimok], ptrim[1][trimok])
            ok[ok] &= trimok
            # if np.sum(trimok) - np.sum(ntrim == 0):
            # print("ntrim not0", np.sum(trimok) - np.sum(ntrim == 0))
        else:
            ok[ok] &= plug.clash_ok(hole.bvh_bb, clashdis, xpos[ok], xeye)
            ptrim = [0], [plug.nres - 1]

        # score everything that didn't clash
        xok = xpos[ok]
        scores = np.zeros((len(xpos), 2))
        scores[ok, 0] = score(iresl, plug, plug, xok, xsym[ok], wts, (*ptrim, *ptrim))
        scores[ok, 1] = score(iresl, plug, hole, xok, xeye[:,], wts, ptrim)

        # record ranges used
        plb = np.zeros(len(scores), dtype="i4")
        pub = np.ones(len(scores), dtype="i4") * (plug.nres - 1)
        if ptrim:
            plb[ok], pub[ok] = ptrim[0], ptrim[1]

        return scores, plb, pub


def filter_redundancy(xforms, plug, scores, **kw):
    args = Bunch(kw)
    nclust = args.max_cluster
    if nclust is None:
        nclust = int(args.beam_size) // 10
    ibest = np.argsort(-scores)
    crd = xforms[ibest[:nclust], None] @ plug.cen[::10, :, None]
    ncen = crd.shape[1]
    crd = crd.reshape(-1, 4 * ncen)
    keep = cookie_cutter(crd, args.rmscut * np.sqrt(ncen))
    print(f"redundancy filter cut {args.rmscut} keep {len(keep)} of {args.max_cluster}")
    return ibest[keep]


def dump_results(args, plug, hole, xforms, scores, ibest, plugeval, **kw):
    t = perf_counter()
    fname_prefix = "plug" if args.out_prefix is None else args.out_prefix
    nout = min(10 if args.nout is None else args.nout, len(ibest))
    for i in range(nout):
        plug.move_to(xforms[ibest[i]])
        ((pscr, hscr),), *lbub = plugeval(xforms[ibest[i]], args.wts.sub(ncontact=0))
        ((pcnt, hcnt),), *lbub = plugeval(
            xforms[ibest[i]], args.wts.sub(rpx=0, ncontact=1)
        )
        fn = fname_prefix + "_%02i.pdb" % i
        print(
            f"{fn} score {scores[ibest[i]]:7.3f} olig: {pscr:7.3f} hole: {hscr:7.3f}",
            f"resi {lbub[0][0]}-{lbub[1][0]} {pcnt:7.0f} {hcnt:7.0f}",
        )
        dump_pdb_from_bodies(fn, [plug], symframes(hole.sym), resbounds=[lbub])
        # dump_pdb_from_bodies(fn + "whole.pdb", [plug], symframes(hole.sym))
    dump_pdb_from_bodies("test_hole.pdb", [hole], symframes(hole.sym))
    return perf_counter() - t


# class PlugWorker(threading.Thread):
#     def __init__(self, xforms, plugeval, nworker, iworker):
#         super().__init__(None, None, None)
#         self.xforms = xforms
#         self.plugeval = plugeval
#         self.nworker = nworker
#         self.iworker = iworker
#
#     def run(self):
#         work = range(self.iworker, len(self.xforms), self.nworker)
#         self.rslt = self.plugeval(self.xforms[work])


def __make_plugs_hier_sample_test__(plug, hole, hscore, **kw):
    args = Bunch(kw)
    sampler = plug_get_sample_hierarchy(plug, hole, hscore)
    sampler = ____PLUG_TEST_SAMPLE_HIERARCHY____(plug, hole, hscore)

    nresl = kw["nresl"]

    for rpx in [0, 1]:
        args.wts = Bunch(plug=1.0, hole=1.0, ncontact=1.0, rpx=rpx)
        plugeval = PlugEvaluator(plug, hole, hscore, **args)
        iresl = 0
        indices, xforms = expand_samples(**args.sub(vars()))
        scores, *resbound, t = evaluate_samples(**args.sub(vars()))
        iroot = np.argsort(-scores)[:10]
        xroot = xforms[iroot]
        sroot = scores[iroot]

        for ibeam in range(6, 27):
            beam_size = 2 ** ibeam
            indices, xforms, scores = iroot, xroot, sroot
            for iresl in range(1, nresl):
                indices, xforms = expand_samples(**args.sub(vars()))
                scores, *resbound, t = evaluate_samples(**args.sub(vars()))
                print(
                    f"rpx {rpx} beam {beam_size:9,}",
                    f"iresl {iresl} ntot {len(scores):11,} nonzero {np.sum(scores > 0):5,}",
                    f"best {np.max(scores)}",
                )
            import _pickle

            fn = "make_plugs_hier_sample_test_rpx_%i_ibeam_%i.pickle" % (rpx, ibeam)
            with open(fn, "wb") as out:
                _pickle.dump((ibeam, iresl, indices, scores), out)
            print()

    assert 0
