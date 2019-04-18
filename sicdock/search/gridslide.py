import numpy as np


def get_cyclic_cyclic_samples(spec, resl=1, max_out_of_plane_angle=10):
    tip = max_out_of_plane_angle
    rots1 = spec.placements1(np.arange(0, 360 // spec.nfold1, resl))
    rots2 = spec.placements2(np.arange(0, 360 // spec.nfold2, resl))
    slideposdn = np.arange(0, -0.001 - tip, -resl)[::-1]
    slideposup = np.arange(resl, tip + 0.001, resl)
    slidenegdn = np.arange(180, 179.999 - tip, -resl)[::-1]
    slidenegup = np.arange(180 + resl, tip + 180.001, resl)
    slides = np.concatenate([slideposdn, slideposup, slidenegdn, slidenegup])
    slides = spec.slide_dir(slides)
    return rots1, rots2, slides


def get_connected_architectures(
    spec, body1, body2, samples, min_contacts=30, contact_dis=8.0
):
    maxsize = len(samples[0]) * len(samples[1]) * len(samples[2])
    npair = np.empty(maxsize, np.int32)
    pos1 = np.empty((maxsize, 4, 4))
    pos2 = np.empty((maxsize, 4, 4))
    nresult = 0
    for x1 in samples[0]:
        body1.move_to(x1)
        for x2 in samples[1]:
            body2.move_to(x2)
            for dirn in samples[2]:
                body1.center()
                d = body1.slide_to(body2, dirn)
                if d < 9e8:
                    npair0 = body1.cen_pair_count(body2, contact_dis)
                    if npair0 >= min_contacts:
                        npair[nresult] = npair0
                        pos1[nresult] = body1.pos
                        pos2[nresult] = body2.pos
                        nresult += 1
    pos1, pos2 = spec.place_along_axes(pos1[:nresult], pos2[:nresult])
    return npair[:nresult], pos1, pos2
