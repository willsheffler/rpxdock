import itertools as it
from sicdock.search import gridslide
import numpy as np
import homog as hm


def hier_start_samples(spec, resl=16, max_out_of_plane_angle=16, nstep=0, **kw):
    tip = max_out_of_plane_angle

    range1 = 180 / spec.nfold1
    range2 = 180 / spec.nfold2
    newresl1 = 2 * range1 / np.ceil(2 * range1 / resl)
    newresl2 = 2 * range2 / np.ceil(2 * range2 / resl)
    angs1 = np.arange(-range1 + newresl1 / 2, range1, newresl1)
    angs2 = np.arange(-range2 + newresl2 / 2, range2, newresl2)
    rots1 = spec.placements1(angs1)
    rots2 = spec.placements2(angs2)

    newresl3 = resl
    angs3 = np.zeros(1)
    if tip > resl / 8:
        newresl3 = 2 * tip / np.ceil(2 * tip / resl)
        angs3 = np.arange(-tip + newresl3 / 2, tip, newresl3)
    slides = np.concatenate([angs3, angs3 + 180])
    slides = spec.slide_dir(slides)

    newresls = np.array([newresl1, newresl2, newresl3])
    angs = (angs1, angs2, angs3)

    return [rots1, rots2, slides], newresls


def hier_expand_samples(spec, pos1, pos2, resls):
    deltas = resls / 2
    assert np.min(deltas) >= 0.1, "deltas should be in degrees"
    deltas = deltas / 180 * np.pi
    n = len(pos1)
    x1 = hm.hrot(spec.axis1, [-deltas[0], +deltas[0]])
    x2 = hm.hrot(spec.axis2, [-deltas[1], +deltas[1]])
    x3 = hm.hrot(spec.axisperp, [-deltas[2], +deltas[2]])
    dirn = (pos2[:, :, 3] - pos1[:, :, 3])[:, :, None]
    dirnorm = np.linalg.norm(dirn, axis=1)
    assert np.min(dirnorm) > 0.9
    # print("hier_expand_samples", n, dirnorm.shape)
    dirn /= dirnorm[:, None]
    newpos1 = np.empty((8 * n, 4, 4))
    newpos2 = np.empty((8 * n, 4, 4))
    newdirn = np.empty((8 * n, 3))
    lb, ub = 0, n
    for x1, x2, x3 in it.product(x1, x2, x3):
        newpos1[lb:ub] = x1 @ pos1
        newpos2[lb:ub] = x2 @ pos2
        newdirn[lb:ub] = (x3 @ dirn)[:, :3].squeeze()
        lb, ub = lb + n, ub + n
    newpos1[:, :3, 3] = 0
    newpos2[:, :3, 3] = 0
    return [newpos1, newpos2, newdirn]


def find_connected_2xCyclic_hier_slide(
    spec,
    body1,
    body2,
    base_resl=16,
    nstep=5,
    base_min_contacts=0,
    prune_frac_sortof=0.875,
    prune_minkeep=1000,
    **kw
):
    assert base_resl > 2, "are you sure?"
    mct = [base_min_contacts]
    mct_update = prune_frac_sortof
    npair, pos = [None] * nstep, [None] * nstep
    samples, newresls = hier_start_samples(spec, resl=base_resl, **kw)
    nsamp = [np.prod([len(s) for s in samples])]
    for i in range(nstep):
        npair[i], pos[i] = gridslide.find_connected_2xCyclic_slide(
            spec, body1, body2, samples, min_contacts=mct[-1], **kw
        )
        if len(npair[i]) is 0:
            return npair[i - 1], pos[i - 1]
        if i + 1 < nstep:
            newresls = newresls / 2
            # print("newresls", newresls)
            samples = hier_expand_samples(spec, *pos[i], newresls)
            nsamp.append(len(samples[0]))

            mct.append(int(np.quantile(npair[i][:, 0], mct_update)))
            # if len(npair[i]) < prune_minkeep:
            #     print("same mct")
            #     mct.append(mct[-1])
            # else:
            #     nmct = npair[i][:, 0].partition(-prune_minkeep)
            #     nmct = npair[i][-prune_minkeep, 0]
            #     qmct = int(np.quantile(npair[i][:, 0], mct_update))
            #     nprint("mct update", nmct, qmct)
            #     mct.append(np.min(nmct, qmct))

    # print("nresult     ", [x.shape[0] for x in npair])
    # print("samps       ", nsamp)
    # print("min_contacts", mct)
    return npair[-1], pos[-1]
