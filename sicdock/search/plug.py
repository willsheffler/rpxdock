import numpy as np
import homog as hm
from sicdock.body import Body
from sicdock.sampling import XformHier_f4
from sicdock.io.io_body import dump_pdb_from_bodies
from sicdock.geom import xform_dist2_split
from sicdock.sym import symframes


def plug_guess_sampling_bounds(plug, hole, cart_samp_resl):
    r0 = max(hole.rg_xy(), 2 * plug.radius_max())

    nr1 = np.ceil(r0 / cart_samp_resl)
    r1 = nr1 * cart_samp_resl

    nr2 = np.ceil(r0 / cart_samp_resl * 2)
    r2 = nr2 * cart_samp_resl / 2

    nh = np.ceil(2 * hole.rg_z() / cart_samp_resl)
    h = nh * cart_samp_resl / 2

    cartub = np.array([+r2, +r2, +h])
    cartlb = np.array([-r2, -r2, -h])
    cartbs = np.array([nr1, nr2, nh], dtype="i")

    return cartlb, cartub, cartbs


def make_plugs(plug, hole, hscore, beam_size=1e6):
    nexpand = max(1, int(beam_size / 64))
    symrot = hm.hrot([0, 0, 1], 360 / int(hole.sym[1:]), degrees=True)
    plugsym = plug.copy()

    cart_samp_resl, ori_samp_resl = hscore.base.attr.xhresl
    cartlb, cartub, cartbs = plug_guess_sampling_bounds(plug, hole, cart_samp_resl)
    xh = XformHier_f4(cartlb, cartub, cartbs, ori_samp_resl)
    print("plug XformHier", xh.size(0), xh.cart_bs, xh.ori_resl, xh.cart_lb, xh.cart_ub)

    Nresl = 4

    for iresl in hscore.iresls()[:Nresl]:

        if iresl == 0:
            indices = np.arange(xh.size(0), dtype="u8")
            mask, xforms = xh.get_xforms(0, indices)
            indices = indices[mask]
        else:
            prev = indices
            xforms_prev = xforms
            indices, xforms = xh.expand_top_N(nexpand, iresl - 1, scores, indices)
            if len(indices) == 0:
                return None

        scores = np.zeros(len(indices))
        for i, xform in enumerate(xforms):
            plug.move_to(xform)
            plugsym.move_to(symrot @ xform)
            if plug.long_axis_z_angle() < 60:
                continue
            if plug.intersects(plugsym) or plug.intersects(hole):
                continue
            pp_scores = hscore.score(iresl, plug, plugsym, 0.001)
            ph_scores = hscore.score(iresl, plug, hole, 0.001)
            # pp_scores = plug.contact_count(plugsym, hscore.maxdis[iresl])
            # ph_scores = plug.contact_count(hole, hscore.maxdis[iresl])
            scores[i] = min(pp_scores, ph_scores)
        print(
            f"iresl {iresl} ntot {len(scores):7,} nonzero {np.sum(scores > 0):5,} max {np.max(scores):8.3f}"
        )

    isort = np.argsort(-scores)
    for i in range(10):
        xform = xforms[isort[i]]
        plug.move_to(xform)
        plugsym.move_to(symrot @ xform)
        hpp = hscore.score(Nresl - 1, plug, plugsym, 0)
        hph = hscore.score(Nresl - 1, plug, hole, 0)
        if scores[isort[i]] <= 0:
            break
        print(f"{i:2} {scores[isort[i]]:7.3f} olig: {hpp:7.3f} hole: {hph:7.3f}")
        plug.move_to(xforms[isort[i]])
        dump_pdb_from_bodies(
            "test_plug_%02i.pdb" % i, [plug, hole.asym_body], symframes(hole.sym)
        )
