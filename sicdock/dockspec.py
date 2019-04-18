import numpy as np
import homog as hm
from sicdock import sym


class DockSpec2CompCage:
    def __init__(self, spec):
        assert len(spec) == 3
        assert spec in "T32 T33 O32 O42 O43 I32 I52 I53".split()
        self.spec = spec
        self.sym = spec[0]
        self.nfold1 = int(spec[1])
        self.nfold2 = int(spec[2])
        assert self.sym in "TOI"
        self.axis1 = sym.axes[self.sym][self.nfold1]
        self.axis2 = sym.axes[self.sym][self.nfold2]
        self.axis2 = sym.axes[self.sym][33] if spec == "T33" else self.axis2
        self.axisangle = hm.angle(self.axis1, self.axis2)
        self.axisperp = hm.hcross(self.axis1, self.axis2)
        self.axisdelta = self.axis2 - self.axis1
        self.axisdelta /= np.linalg.norm(self.axisdelta)
        self.orig1 = hm.align_vector([0, 0, 1], self.axis1)
        self.orig2 = hm.align_vector([0, 0, 1], self.axis2)
        self.symframes_ = sym.frames[self.sym]
        self.axis1_second = sym.axes_second[self.sym][self.nfold1]
        self.axis2_second = sym.axes_second[self.sym][self.nfold2]
        self.to_neighbor_olig1 = sym.to_neighbor_olig[self.sym][self.nfold1]
        self.to_neighbor_olig2 = sym.to_neighbor_olig[self.sym][self.nfold2]
        if spec == "T33":
            self.axis2_second = sym.axes_second[self.sym][33]
            self.to_neighbor_olig2 = sym.to_neighbor_olig[self.sym][33]

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
