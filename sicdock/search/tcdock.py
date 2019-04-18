import numpy as np
import homog as hm
from sicdock.sym import symaxes, symframes


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


def get_connected_architectures(
    body1, body2, arch, resl=1, maxtip=10, minpairs=30, maxpairdis=8.0
):
    samp1 = arch.placements1(range(0, 360 // arch.nfold1, resl))
    samp2 = arch.placements2(range(0, 360 // arch.nfold2, resl))
    slideangpos = list(range(-maxtip, maxtip + 1, resl))
    slideangneg = list(range(180 - maxtip, maxtip + 181, resl))
    samp3 = arch.slide_dir(slideangpos + slideangneg)

    maxsize = len(samp1) * len(samp2) * len(samp3)
    npair = np.empty(maxsize, np.int32)
    pos1 = np.empty((maxsize, 4, 4))
    pos2 = np.empty((maxsize, 4, 4))
    nresult = 0
    for x1 in samp1:
        body1.move_to(x1)
        for x2 in samp2:
            body2.move_to(x2)
            for dirn in samp3:
                body1.center()
                d = body1.slide_to(body2, dirn)
                if d < 9e8:
                    npair0 = body1.cen_pair_count(body2, maxpairdis)
                    if npair0 >= minpairs:
                        npair[nresult] = npair0
                        pos1[nresult] = body1.pos
                        pos2[nresult] = body2.pos
                        nresult += 1
    pos1, pos2 = arch.place_along_axes(pos1[:nresult], pos2[:nresult])
    return npair[:nresult], pos1, pos2
