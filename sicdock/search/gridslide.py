import numpy as np
import sicdock.sampling.orientations as ori
from homog.quat import quat_to_xform


def samples_1xMonomer_orientations(resl):
    quats = ori.quaternion_set_with_covering_radius_degrees(resl)[0]
    return quat_to_xform(quats)


def samples_1xCyclic(spec, resl=1):
    return spec.placements(np.arange(0, 360 // spec.nfold, resl))


def samples_2xCyclic(spec, resl=1, max_out_of_plane_angle=10):
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


def find_connected_1xCyclic(spec, body, samples, min_contacts=30, contact_dis=8.0):
    body2 = body.copy()  # shallow copy except pos
    samples_second = spec.placements_second(samples)
    maxsize = len(samples)
    npair = np.empty(maxsize, np.int32)
    pos = np.empty((maxsize, 4, 4))
    dslide = np.empty(maxsize)
    nresult, nhit = 0, 0
    dirn = spec.slide_dir()
    for x1, x2 in zip(samples, samples_second):
        body.move_to(x1)
        body2.move_to(x2)
        d = body.slide_to(body2, dirn)
        if d < 9e8:
            nhit += 1
            npair0 = body.cen_pair_count(body2, contact_dis)
            if npair0 >= min_contacts:
                npair[nresult] = npair0
                pos[nresult] = body.pos
                dslide[nresult] = d
                nresult += 1
    assert nhit == maxsize
    pos = spec.place_along_axis(pos[:nresult], dslide[:nresult])
    return npair[:nresult], pos


def find_connected_2xCyclic(
    spec, body1, body2, samples, min_contacts=30, contact_dis=8.0
):
    maxsize = len(samples[0]) * len(samples[1]) * len(samples[2])
    npair = np.empty(maxsize, np.int32)
    pos1 = np.empty((maxsize, 4, 4))
    pos2 = np.empty((maxsize, 4, 4))
    nresult, nhit = 0, 0
    for x1 in samples[0]:
        body1.move_to(x1)
        for x2 in samples[1]:
            body2.move_to(x2)
            for dirn in samples[2]:
                body1.center()
                d = body1.slide_to(body2, dirn)
                if d < 9e8:
                    nhit += 1
                    npair0 = body1.cen_pair_count(body2, contact_dis)
                    if npair0 >= min_contacts:
                        npair[nresult] = npair0
                        pos1[nresult] = body1.pos
                        pos2[nresult] = body2.pos
                        nresult += 1
    assert nhit == maxsize
    pos1, pos2 = spec.place_along_axes(pos1[:nresult], pos2[:nresult])
    return npair[:nresult], pos1, pos2


def find_connected_monomer_to_cyclic(spec, body, samples, min_contacts, contact_dis):
    body2 = body.copy()  # shallow copy except pos
    samples_second = spec.placements_second(samples)
    maxsize = len(samples)
    npair = np.empty(maxsize, np.int32)
    pos = np.empty((maxsize, 4, 4))
    dslide = np.empty(maxsize)
    nresult, nhit = 0, 0
    dirn = spec.slide_dir()
    for x1, x2 in zip(samples, samples_second):
        body.move_to(x1)
        body2.move_to(x2)
        d = body.slide_to(body2, dirn)
        if d < 9e8:
            nhit += 1
            npair0 = body.cen_pair_count(body2, contact_dis)
            if npair0 >= min_contacts:
                npair[nresult] = npair0
                pos[nresult] = body.pos
                dslide[nresult] = d
                nresult += 1
    assert nhit == maxsize
    pos = spec.place_along_axis(pos[:nresult], dslide[:nresult])
    return npair[:nresult], pos
