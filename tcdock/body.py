from time import perf_counter
import numpy as np
import homog as hm

import tcdock.rosetta as ros
from tcdock.bvh import (
    bvh_create,
    bvh_slide,
    bvh_collect_pairs,
    bvh_isect,
    naive_min_dist,
    bvh_min_dist,
    bvh_print,
)
from tcdock.io import pdb_format_atom


_CLASH_radius = 1.7


class Body:
    def __init__(self, pdbfile, sym=None, **kw):
        self.pdbfile = pdbfile
        self.sym = sym
        self.pose = ros.pose_from_file(pdbfile)

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
            print(self.coord.shape, newcoord.shape)
            for i in range(1, nfold):
                self.pos = hm.hrot([0, 0, 1], 360.0 * i / nfold)
                newcoord[i * n :][:n] = self.positioned_coord()
            self.coord = newcoord
        assert len(self.seq) == len(self.coord)
        assert len(self.ss) == len(self.coord)
        assert len(self.chain) == len(self.coord)

        self.stub = ros.get_bb_stubs(self.coord)
        self.bvh_bb = bvh_create(self.coord[..., :3].reshape(-1, 3))
        cen = self.stub[:, :3, 3]
        which_cen = ~np.isin(self.seq, ["G", "C", "P"])
        self.bvh_cen = bvh_create(cen, which_cen)
        self.pos = np.eye(4)
        self.pair_buf = np.empty((10000, 2), dtype="i4")

    def move_by(self, x):
        self.pos = x @ self.pos

    def move_to(self, x):
        self.pos = x

    def center(self):
        self.pos[:3, 3] = 0

    def slide_to(self, other, dirn):
        dirn = np.array(dirn, dtype=np.float64)
        dirn /= np.linalg.norm(dirn)
        radius = _CLASH_radius
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

    def intersects(self, other, mindis):
        return bvh_isect(self.bvh_bb, other.bvh_bb, self.pos, other.pos, mindis)

    def distance_to(self, other):
        return bvh_min_dist(self.bvh_bb, other.bvh_bb, self.pos, other.pos)

    def positioned_coord(self):
        return (self.pos @ self.coord[..., None]).squeeze()

    def positioned_cen(self):
        cen = self.stub[..., 3]
        return (self.pos @ cen[..., None]).squeeze()

    def cen_pairs(self, other, maxdis):
        buf = self.pair_buf
        n = bvh_collect_pairs(
            self.bvh_cen, other.bvh_cen, self.pos, other.pos, maxdis, buf
        )
        return self.pair_buf[:n]

    def dump_pdb(self, fname):
        s = ""
        ia = 0
        for i in range(len(self.coord)):
            crd = self.positioned_coord()
            cen = self.positioned_cen()
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


if __name__ == "__main__":

    pdbs = [
        "/home/sheffler/scaffolds/C2_best/C2_3hm4_1.pdb",
        "/home/sheffler/scaffolds/C2_best/C2_3l9f_1.pdb",
        "/home/sheffler/scaffolds/C2_best/C2_3lzl_1.pdb",
        "/home/sheffler/scaffolds/C2_best/C2_3u80_1.pdb",
        "/home/sheffler/scaffolds/C2_best/C2_jf21_1.pdb",
        "/home/sheffler/scaffolds/C2_best/C2_pb21_1.pdb",
        "/home/sheffler/scaffolds/C2_best/C2_yl21_1.pdb",
        "/home/sheffler/scaffolds/C3_best/C3_1hfo_1.pdb",
        "/home/sheffler/scaffolds/C3_best/C3_1nza_1.pdb",
        "/home/sheffler/scaffolds/C3_best/C3_1ufy_1.pdb",
        "/home/sheffler/scaffolds/C3_best/C3_1wa3_1.pdb",
        "/home/sheffler/scaffolds/C3_best/C3_1woz_1.pdb",
        "/home/sheffler/scaffolds/C3_best/C3_1wvt_1.pdb",
        "/home/sheffler/scaffolds/C3_best/C3_1wy1_1.pdb",
        "/home/sheffler/scaffolds/C3_best/C3_2c0a_1.pdb",
        "/home/sheffler/scaffolds/C3_best/C3_2yw3_1.pdb",
        "/home/sheffler/scaffolds/C3_best/C3_3e6q_1.pdb",
        "/home/sheffler/scaffolds/C3_best/C3_3ftt_1.pdb",
        "/home/sheffler/scaffolds/C3_best/C3_3fuy_1.pdb",
        "/home/sheffler/scaffolds/C3_best/C3_3fwu_1.pdb",
        "/home/sheffler/scaffolds/C3_best/C3_3k6a_1.pdb",
        "/home/sheffler/scaffolds/C3_best/C3_3n79_1.pdb",
        "/home/sheffler/scaffolds/C3_best/C3_3nz2_3.pdb",
        "/home/sheffler/scaffolds/C3_best/C3_4e38_1.pdb",
        "/home/sheffler/scaffolds/C3_best/C3_jf31_1.pdb",
        "/home/sheffler/scaffolds/C3_best/C3_pb31_1.pdb",
    ]
    body2 = Body("/home/sheffler/scaffolds/C2_best/C2_3hm4_1.pdb", "C2")
    body3 = Body("/home/sheffler/scaffolds/C3_best/C3_1nza_1.pdb", "C3")

    # pymol sel
    # p = body2.cen_pairs(body3, 8.0)
    # s1 = "+".join(str(body2.resno[i]) for i in np.unique(p[:, 0]))
    # s2 = "+".join(str(body3.resno[i]) for i in np.unique(p[:, 1]))
    # print("select s1=body2 and resi", s1)
    # print("select s2=body3 and resi", s2)

    samp1 = range(0, 180, 3)
    samp2 = range(0, 120, 3)
    samp3 = range(-30, 31, 3)
    samp3 = [[np.cos(d / 180 * np.pi), 0, np.sin(d / 180 * np.pi)] for d in samp3]

    r1 = hm.hrot([0, 0, 1], 1, degrees=True)
    best, bestpos = -9e9, None
    t = perf_counter()
    totslide, totpair = 0, 0
    nsamp, nhit = 0, 0
    for a1 in samp1:
        for a2 in samp2:
            for dirn in samp3:
                body2.center()
                body3.center()
                d = body2.slide_to(body3, dirn)
                nsamp += 1
                if d < 9e8:
                    nhit += 1
                    p = body2.cen_pairs(body3, 8.0)
                    if len(p) > best:
                        best = len(p)
                        bestpos = body2.pos.copy(), body3.pos.copy()
            body3.move_by(r1)
        body2.move_by(r1)
    t = perf_counter() - t
    print("best", best, "time", t, "rate", nsamp / t, "hitfrac", nhit / nsamp)
    print(bestpos[0])
    print(bestpos[1])
    body2.move_to(bestpos[0])
    body3.move_to(bestpos[1])
    body2.dump_pdb("body2.pdb")
    body3.dump_pdb("body3.pdb")


# select body2 and resi 23+25+146+148+149+150

# select s3 = body3 and resi 72+73+78+82+83+86+87+283
