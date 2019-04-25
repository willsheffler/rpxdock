import copy

import numpy as np
import homog as hm

import sicdock.rosetta as ros
from sicdock.bvh import (
    bvh_create,
    bvh_slide,
    bvh_collect_pairs,
    bvh_isect,
    naive_min_dist,
    bvh_min_dist,
    bvh_print,
    bvh_count_pairs,
)
from sicdock import motif

_CLASH_RADIUS = 1.75


class Body:
    def __init__(self, pdb, sym="C1", which_ss="HE", posecache=False, **kw):
        if isinstance(pdb, str):
            self.pdbfile = pdb
            if posecache:
                self.pose = ros.get_pose_cached(pdb)
            else:
                self.pose = ros.pose_from_file(pdb)
        else:
            self.pose = pdb

        if isinstance(sym, int):
            sym = "C%i" % sym
        self.sym = sym
        self.nfold = int(sym[1:])
        self.seq = np.array(list(self.pose.sequence()))
        self.ss = np.array(list(self.pose.secstruct()))
        self.coord = ros.get_bb_coords(self.pose)
        self.chain = np.repeat(0, self.seq.shape[0])
        self.resno = np.arange(len(self.seq))

        if sym and sym[0] == "C":
            n = self.coord.shape[0]
            nfold = int(sym[1:])
            self.seq = np.array(list(nfold * self.pose.sequence()))
            self.ss = np.array(list(nfold * self.pose.secstruct()))
            self.chain = np.repeat(range(nfold), n)
            self.resno = np.tile(range(n), nfold)
            newcoord = np.empty((nfold * n,) + self.coord.shape[1:])
            newcoord[:n] = self.coord
            # print(self.coord.shape, newcoord.shape)
            for i in range(1, nfold):
                self.pos = hm.hrot([0, 0, 1], 360.0 * i / nfold)
                newcoord[i * n :][:n] = self.positioned_coord()
            self.coord = newcoord
        assert len(self.seq) == len(self.coord)
        assert len(self.ss) == len(self.coord)
        assert len(self.chain) == len(self.coord)

        self.stub = motif.bb_stubs(self.coord)
        self.bvh_bb = bvh_create(self.coord[..., :3].reshape(-1, 3))
        self.allcen = self.stub[:, :, 3]
        which_cen = np.repeat(False, len(self.ss))
        for ss in "EHL":
            if ss in which_ss:
                which_cen |= self.ss == ss
        which_cen &= ~np.isin(self.seq, ["G", "C", "P"])
        self.bvh_cen = bvh_create(self.allcen[:, :3], which_cen)
        self.cen = self.allcen[which_cen]
        self.pos = np.eye(4)
        self.pair_buf = np.empty((10000, 2), dtype="i4")

    def com(self):
        return self.pos @ self.bvh_bb.com()

    def rg(self):
        d = self.cen - self.com()
        return np.sqrt(np.sum(d ** 2) / len(d))

    def radius_max(self):
        return np.max(self.cen - self.com())

    def rg_xy(self):
        d = self.cen[:, :2] - self.com()[:2]
        rg = np.sqrt(np.sum(d ** 2) / len(d))
        return rg

    def radius_xy_max(self):
        return np.max(self.cen[:, :2] - self.com()[:2])

    def move_by(self, x):
        self.pos = x @ self.pos
        return self

    def move_to(self, x):
        self.pos = x.copy()
        return self

    def move_to_center(self):
        self.pos[:3, 3] = 0
        return self

    def slide_to(self, other, dirn, radius=_CLASH_RADIUS):
        dirn = np.array(dirn, dtype=np.float64)
        dirn /= np.linalg.norm(dirn)
        delta = bvh_slide(self.bvh_bb, other.bvh_bb, self.pos, other.pos, radius, dirn)
        if delta < 9e8:
            self.pos[:3, 3] += delta * dirn

            # print(self.coord.shape, other.coord.shape)
            # u = self.positioned_coord().reshape(-1, 1, 4)
            # v = other.positioned_coord().reshape(1, -1, 4)
            # mind = np.linalg.norm(u - v, axis=2)
            # d1 = bvh_min_dist(self.bvh_bb, other.bvh_bb, self.pos, other.pos)
            # d2 = naive_min_dist(self.bvh_bb, other.bvh_bb, self.pos, other.pos)
            # print("slide_to sanity check mindis", np.min(mind), d1)
            # d, i, j = d1
            # print("mindis", d, d2)
            # p1 = self.coord[..., :3].reshape(-1, 3)[i]
            # p2 = other.coord[..., :3].reshape(-1, 3)[j]
            # p1 = self.pos @ self.coord.reshape(-1, 4)[i]
            # p2 = other.pos @ other.coord.reshape(-1, 4)[j]
            # print(p1, p2, np.linalg.norm(p1 - p2))
        # else:
        # print("MISS")

        return delta

    def intersects(self, other, mindis=2 * _CLASH_RADIUS):
        return bvh_isect(self.bvh_bb, other.bvh_bb, self.pos, other.pos, mindis)

    def distance_to(self, other):
        return bvh_min_dist(self.bvh_bb, other.bvh_bb, self.pos, other.pos)

    def positioned_coord(self, asym=False):
        n = len(self.coord) // self.nfold if asym else len(self.coord)
        return (self.pos @ self.coord[:n, :, :, None]).squeeze()

    def positioned_cen(self, asym=False):
        n = len(self.stub) // self.nfold if asym else len(self.stub)
        cen = self.stub[:n, :, 3]
        return (self.pos @ cen[..., None]).squeeze()

    def cen_pairs(self, other, maxdis, buf=None):
        if not buf:
            buf = self.pair_buf
        n = bvh_collect_pairs(
            self.bvh_cen, other.bvh_cen, self.pos, other.pos, maxdis, buf
        )
        return buf[:n]

    def cen_pair_count(self, other, maxdis):
        return bvh_count_pairs(self.bvh_cen, other.bvh_cen, self.pos, other.pos, maxdis)

    def dump_pdb(self, fname, asym=False):
        from sicdock.io import pdb_format_atom

        s = ""
        ia = 0
        crd = self.positioned_coord(asym=asym)
        cen = self.positioned_cen(asym=asym)
        for i in range(len(crd)):
            c = self.chain[i]
            j = self.resno[i]
            aa = self.seq[i]
            s += pdb_format_atom(ia=ia + 0, ir=j, rn=aa, xyz=crd[i, 0], c=c, an="N")
            s += pdb_format_atom(ia=ia + 1, ir=j, rn=aa, xyz=crd[i, 1], c=c, an="CA")
            s += pdb_format_atom(ia=ia + 2, ir=j, rn=aa, xyz=crd[i, 2], c=c, an="C")
            s += pdb_format_atom(ia=ia + 3, ir=j, rn=aa, xyz=crd[i, 3], c=c, an="O")
            s += pdb_format_atom(ia=ia + 4, ir=j, rn=aa, xyz=crd[i, 4], c=c, an="CB")
            s += pdb_format_atom(ia=ia + 5, ir=j, rn=aa, xyz=cen[i], c=c, an="CEN")
            ia += 6

        with open(fname, "w") as out:
            out.write(s)

    def copy(self):
        b = copy.copy(self)
        b.pos = np.eye(4)  # mutable state can't be same ref as orig
        assert b.pos is not self.pos
        assert b.coord is self.coord
        return b
