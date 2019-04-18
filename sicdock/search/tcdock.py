import numpy as np
import homog as hm
from sicdock.sym import symaxes


class Arch:
    def __init__(self, arch):
        assert len(arch) == 3
        self.arch = arch
        self.sym = arch[0]
        self.nfold1 = int(arch[1])
        self.nfold2 = int(arch[2])
        assert self.sym in "TOI"
        self.axis1 = symaxes[self.sym][self.nfold1]
        self.axis2 = symaxes[self.sym][self.nfold2]
        self.orig1 = hm.align_vector([0, 0, 1], self.axis1)
        self.orig2 = hm.align_vector([0, 0, 1], self.axis2)

    def align_bodies(self, body1, body2):
        body1.move_to(self.orig1)
        body2.move_to(self.orig2)

    def slide_dir(self, angles):
        dcen = hm.hnormalized(self.axis2 - self.axis1)
        perp = hm.hcross(self.axis1, self.axis2)
        dirs = hm.hrot(perp, angles) @ dcen
        return dirs[..., :3]

    def placements1(self, angles):
        place1 = hm.hrot(self.axis1, angles, degrees=1) @ self.orig1
        place2 = place1 @ hm.hrot([1, 0, 0], 180)
        return np.concatenate([place1, place2])

    def placements2(self, angles):
        return hm.hrot(self.axis2, angles, degrees=1) @ self.orig2

    def place_bodies(self, pos1, pos2):
        pos1, pos2 = pos1.copy(), pos2.copy()
        offset = pos2[:, 3] - pos1[:, 3]
        cang = hm.angle(self.axis1, self.axis2)
        cdis = np.linalg.norm(offset)

        aofst = hm.line_angle(offset, self.axis2 - self.axis1)
        print("aofst", aofst * 180 / np.pi)
        aang = (np.pi - cang) / 2.0 - aofst
        bang = (np.pi - cang) / 2.0 + aofst
        adis = cdis * np.sin(aang) / np.sin(cang)
        bdis = cdis * np.sin(bang) / np.sin(cang)
        print("a", adis, aang * 180 / np.pi)
        print("b", bdis, bang * 180 / np.pi)
        print("c", cdis, cang * 180 / np.pi)
        pos1[:3, 3] = self.axis1[:3] * adis
        pos2[:3, 3] = self.axis2[:3] * bdis
        return (pos1, pos2)


def tcdock(body1, body2, arch, resl=1, maxtip=10):
    samp1 = arch.placements1(range(0, 360 // arch.nfold1, resl))
    samp2 = arch.placements2(range(0, 360 // arch.nfold2, resl))
    samp3 = arch.slide_dir(range(-maxtip, maxtip + 1, resl))

    maxpairdis = 8.0
    best, bestpos = -9e9, None
    for pos1 in samp1:
        body1.move_to(pos1)
        for pos2 in samp2:
            body2.move_to(pos2)
            for dirn in samp3:
                body1.center()
                d = body1.slide_to(body2, dirn)
                if d < 9e8:
                    p = body1.cen_pairs(body2, maxpairdis)
                    if len(p) > best:
                        best = len(p)
                        bestpos = body1.pos.copy(), body2.pos.copy()

    return best, bestpos
