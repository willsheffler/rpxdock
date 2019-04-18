import numpy as np
import homog as hm
from sicdock.sym import symaxes, symframes, symaxes_second, sym_to_neighbor_olig


class Architecture:
    def __init__(self, arch):
        assert len(arch) == 3
        self.arch = arch
        self.sym = arch[0]
        self.nfold1 = int(arch[1])
        self.nfold2 = int(arch[2])
        assert self.sym in "TOI"
        self.axis1 = symaxes[self.sym][self.nfold1]
        self.axis2 = symaxes[self.sym][self.nfold2]
        self.axisangle = hm.angle(self.axis1, self.axis2)
        self.axisperp = hm.hcross(self.axis1, self.axis2)
        self.axisdelta = self.axis2 - self.axis1
        self.axisdelta /= np.linalg.norm(self.axisdelta)
        self.orig1 = hm.align_vector([0, 0, 1], self.axis1)
        self.orig2 = hm.align_vector([0, 0, 1], self.axis2)
        self.symframes_ = symframes[self.sym]
        self.axis1_second = symaxes_second[self.sym][self.nfold1]
        self.axis2_second = symaxes_second[self.sym][self.nfold2]
        self.to_neighbor_olig1 = sym_to_neighbor_olig[self.sym][self.nfold1]
        self.to_neighbor_olig2 = sym_to_neighbor_olig[self.sym][self.nfold2]

    def align_bodies(self, body1, body2):
        body1.move_to(self.orig1)
        body2.move_to(self.orig2)

    def slide_dir(self, angles):
        dcen = self.axisdelta
        dirs = hm.hrot(self.axisperp, angles) @ dcen
        return dirs[..., :3]

    def placements1(self, angles):
        place1 = hm.hrot(self.axis1, angles, degrees=1) @ self.orig1
        place2 = place1 @ hm.hrot([1, 0, 0], 180)
        return np.concatenate([place1, place2])

    def placements2(self, angles):
        return hm.hrot(self.axis2, angles, degrees=1) @ self.orig2

    def symframes(self, cellspacing=None, radius=None):
        return self.symframes_

    def move_to_canonical_unit(self, pos1, pos2):
        origshape = pos1.shape
        pos1 = pos1.reshape(-1, 4, 4).copy()
        pos2 = pos2.reshape(-1, 4, 4).copy()
        xcen = self.symframes_[:, None] @ pos1[:, :, 3, None]
        xcen += self.symframes_[:, None] @ pos2[:, :, 3, None]
        tgt = self.axis1 + self.axis2
        xdot = np.sum(xcen.squeeze() * tgt, axis=2)
        x = self.symframes_[np.argmax(xdot, axis=0)]
        return (x @ pos1).reshape(origshape), (x @ pos2).reshape(origshape)

    def place_along_axes(self, pos1, pos2):
        origshape = pos1.shape
        pos1 = pos1.reshape(-1, 4, 4).copy()
        pos2 = pos2.reshape(-1, 4, 4).copy()
        offset = pos2[:, :3, 3] - pos1[:, :3, 3]
        cang = self.axisangle
        cdis = np.linalg.norm(offset, axis=1)

        offset_norm = offset / cdis[:, None]
        dot1 = hm.hdot(offset_norm, self.axis1)
        dot2 = hm.hdot(offset_norm, self.axis2)
        aofst = np.sign(dot1 + dot2) * hm.angle(offset_norm, self.axisdelta)
        aang = (np.pi - cang) / 2.0 - aofst
        bang = (np.pi - cang) / 2.0 + aofst

        adis = cdis * np.sin(aang) / np.sin(cang)
        bdis = cdis * np.sin(bang) / np.sin(cang)
        newt1 = self.axis1[:3] * adis[:, None]
        newt2 = self.axis2[:3] * bdis[:, None]
        newoffset = newt2 - newt1

        swap = hm.hdot(offset, newoffset) < 0
        newt1[swap] *= -1
        newt2[swap] *= -1
        assert np.allclose(newt2 - newt1, offset[:, :3])
        pos1[:, :3, 3] = newt1
        pos2[:, :3, 3] = newt2

        return (pos1.reshape(origshape), pos2.reshape(origshape))


def get_cyclic_cyclic_samples(arch, resl=1, max_out_of_plane_angle=10):
    tip = max_out_of_plane_angle
    rots1 = arch.placements1(np.arange(0, 360 // arch.nfold1, resl))
    rots2 = arch.placements2(np.arange(0, 360 // arch.nfold2, resl))
    slideposdn = np.arange(0, -0.001 - tip, -resl)[::-1]
    slideposup = np.arange(resl, tip + 0.001, resl)
    slidenegdn = np.arange(180, 179.999 - tip, -resl)[::-1]
    slidenegup = np.arange(180 + resl, tip + 180.001, resl)
    slides = np.concatenate([slideposdn, slideposup, slidenegdn, slidenegup])
    slides = arch.slide_dir(slides)
    return rots1, rots2, slides


def get_connected_architectures(
    arch, body1, body2, samples, min_contacts=30, contact_dis=8.0
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
    pos1, pos2 = arch.place_along_axes(pos1[:nresult], pos2[:nresult])
    return npair[:nresult], pos1, pos2
