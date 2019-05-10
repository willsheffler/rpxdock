import threading
from time import perf_counter
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import homog as hm
from sicdock.body import Body
from sicdock.sampling import XformHier_f4
from sicdock.io.io_body import dump_pdb_from_bodies
from sicdock.geom import xform_dist2_split
from sicdock.sym import symframes
from sicdock.bvh import bvh_isect, bvh_count_pairs, bvh_isect_vec, bvh_count_pairs_vec

xeye = np.eye(4, dtype="f4")


def plug_guess_sampling_bounds(plug, hole, xhresl):
    cart_samp_resl, ori_samp_resl = xhresl

    r0 = max(hole.rg_xy(), 2 * plug.radius_max())

    nr1 = np.ceil(r0 / cart_samp_resl)
    r1 = nr1 * cart_samp_resl

    nr2 = np.ceil(r0 / cart_samp_resl * 2)
    r2 = nr2 * cart_samp_resl / 2

    nh = np.ceil(4 * hole.rg_z() / cart_samp_resl)
    h = nh * cart_samp_resl / 2

    cartub = np.array([+r2, +r2, +h])
    cartlb = np.array([-r2, -r2, -h])
    cartbs = np.array([nr2, nr2, nh], dtype="i")

    return cartlb, cartub, cartbs, ori_samp_resl


def ____PLUG_TEST_SAMPLING_BOUNDS____(plug, hole, xhresl):
    r, ori_samp_resl = xhresl
    cartub = np.array([6 * r, r, r])
    cartlb = np.array([-6 * r, 0, 0])
    cartbs = np.array([12, 1, 1], dtype="i")
    return cartlb, cartub, cartbs, ori_samp_resl * 1.0


def make_plugs(
    plug,
    hole,
    hscore,
    beam_size=1e4,
    w_plug=1.0,
    w_hole=1.0,
    wcontact=0.001,
    nworker=8,
    nresl=None,
    **kw,
):

    do_main = False
    do_threads = True

    assert nworker > 0
    nresl = len(hscore.hier) if nresl is None else nresl

    # xhargs = ____PLUG_TEST_SAMPLING_BOUNDS____(plug, hole, hscore.base.attr.xhresl)
    xhargs = plug_guess_sampling_bounds(plug, hole, hscore.base.attr.xhresl)
    xh = XformHier_f4(*xhargs)
    assert xh.sanity_check(), "bad xform hierarchy"
    print("plug XformHier", xh.size(0), xh.cart_bs, xh.ori_resl, xh.cart_lb, xh.cart_ub)

    tsamp, tmain, tthread, texec, tdump, ttot = [0] * 5 + [perf_counter()]

    # executor = ThreadPoolExecutor(max_workers=nworker)
    evaluator = PlugEvaluator(plug, hole, hscore, wcontact=wcontact)

    for iresl in range(nresl):
        evaluator.iresl = iresl

        t = perf_counter()
        if iresl == 0:
            indices = np.arange(xh.size(0), dtype="u8")
            mask, xforms = xh.get_xforms(0, indices)
            indices = indices[mask]
        else:
            nexpand = max(1, int(beam_size / 64))
            indices, xforms = xh.expand_top_N(nexpand, iresl - 1, scores, indices)
            if len(indices) == 0:
                print("FAIL at", iresl)
                break
        tsamp += perf_counter() - t

        # ################## manual threads
        t = perf_counter()
        if do_threads:
            workers = [Worker(xforms, evaluator, nworker, i) for i in range(nworker)]
            [w.start() for w in workers]
            [w.join() for w in workers]
            scores = np.empty(len(indices))
            for i, w in enumerate(workers):
                scores[i::nworker] = np.minimum(
                    w_plug * w.rslt[:, 0], w_hole * w.rslt[:, 1]
                )
        tthread += perf_counter() - t

        if do_main:
            t = perf_counter()
            mscores = np.empty(len(indices))
            for i, xform in enumerate(xforms):
                s = evaluator(xform)[0]
                mscores[i] = np.minimum(w_plug * s[0], w_hole * s[1])
            tmain = perf_counter() - t
            if do_threads:
                assert np.allclose(mscores, scores)
            scores = mscores

        # ################## executor
        # t = perf_counter()
        # scores2 = np.array(
        #     [x for x in executor.map(evaluator, np.split(xforms, nworker))]
        # )
        # escores = np.sum(np.concatenate(scores2), axis=1)
        # texec = perf_counter() - t
        # assert np.allclose(escores, tscores)

        print(
            f"iresl {iresl} ntot {len(scores):7,} nonzero {np.sum(scores > 0):5,} max {np.max(scores):8.3f}"
        )
        ########### dump top 10 #############
        t = perf_counter()
        isort = np.argsort(-scores)
        for i in range(1 if iresl + 1 < nresl else 10):
            if scores[isort[i]] <= 0:
                break
            hpp, hph = evaluator(xforms[isort[i]], wcontact=0)[0]
            print(
                f"stage {iresl} {i:2} score {scores[isort[i]]:7.3f}",
                f"olig: {hpp:7.3f} hole: {hph:7.3f}",
            )
            plug.move_to(xforms[isort[i]])
            dump_pdb_from_bodies(
                "test_plug_%i_%02i.pdb" % (iresl, i), [plug], symframes(hole.sym)
            )
        tdump += perf_counter() - t
    dump_pdb_from_bodies("test_hole.pdb", [hole], symframes(hole.sym))

    # executor.shutdown(wait=True)

    ttot = perf_counter() - ttot
    print("=" * 80)
    print(
        f"ttot {ttot:7.3f} tthread {tthread:7.3f} tdump {tdump:7.3f} tmain {tmain:7.3f}",
        f"texec {texec:7.3f} tsamp {tsamp:7.3f} tmain/tthread {tmain/tthread:7.3f}",
    )
    print("=" * 80)


class Worker(threading.Thread):
    def __init__(self, xforms, evaluator, nworker, iworker):
        super().__init__(None, None, None)
        self.xforms = xforms
        self.evaluator = evaluator
        self.nworker = nworker
        self.iworker = iworker

    def run(self):
        work = range(self.iworker, len(self.xforms), self.nworker)
        self.rslt = self.evaluator(self.xforms[work])


class PlugEvaluator:
    def __init__(self, plug, hole, hscore, wcontact=0.001):
        self.plug = plug.copy()
        self.plugsym = plug.copy()
        self.hole = hole
        self.hscore = hscore
        self.symrot = hm.hrot([0, 0, 1], 360 / int(hole.sym[1:]), degrees=True)
        self.iresl = 0
        self.wcontact = wcontact

    def __call__(self, xforms, wcontact=None):
        wcontact = self.wcontact if wcontact is None else wcontact
        xforms = xforms.reshape(-1, 4, 4)
        plug, hole, iresl, hscore = self.plug, self.hole, self.iresl, self.hscore
        xsym = self.symrot @ xforms
        ok = np.abs((xforms @ plug.pcavecs[0])[:, 2]) <= 0.5
        ok[ok] &= ~bvh_isect_vec(plug.bvh_bb, plug.bvh_bb, xforms[ok], xsym[ok], 3.5)
        ok[ok] &= ~bvh_isect_vec(plug.bvh_bb, hole.bvh_bb, xforms[ok], xeye[:,], 3.5)
        xok = xforms[ok]
        score = np.zeros((len(xforms), 2))
        score[ok, 0] = hscore.scorepos(iresl, plug, plug, xok, xsym[ok], wcontact)
        score[ok, 1] = hscore.scorepos(iresl, plug, hole, xok, xeye[:,], wcontact)
        return score
