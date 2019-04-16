import numpy as np

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
    def __init__(self, pdbfile, sym=1, **kw):
        self.pdbfile = pdbfile
        self.sym = sym
        self.pose = ros.pose_from_file(pdbfile)
        self.seq = np.array(list(self.pose.sequence()))
        self.ss = np.array(list(self.pose.secstruct()))
        self.coord = ros.get_bb_coords(self.pose)

        # self.coord = self.coord[:2]
        # self.seq = self.seq[:2]

        self.stub = ros.get_bb_stubs(self.coord)
        self.bvh_bb = bvh_create(self.coord[..., :3].reshape(-1, 3))

        # print("self.coord", self.coord)
        # print("bvh_print")
        # import sys

        # sys.stdout.flush()
        # bvh_print(self.bvh_bb)

        cen = self.stub[:, :3, 3]
        which_cen = np.isin(self.seq, ["G", "C", "P"])
        self.bvh_cen = bvh_create(cen, which_cen)
        self.pos = np.eye(4)

        # print("self.coord", cen)
        # print("bvh_print")
        # sys.stdout.flush()
        # bvh_print(self.bvh_cen)

    def slide_to(self, other, dirn):
        dirn = np.array(dirn, dtype=np.float64)
        print(dirn, dirn.dtype, np.linalg.norm(dirn))
        dirn /= np.linalg.norm(dirn)
        radius = _CLASH_radius
        delta = bvh_slide(self.bvh_bb, other.bvh_bb, self.pos, other.pos, radius, dirn)
        if delta < 9e8:
            self.pos[:3, 3] += delta * dirn

            # print(self.coord.shape, other.coord.shape)
            # u = self.positioned_coords().reshape(-1, 1, 4)
            # v = other.positioned_coords().reshape(1, -1, 4)
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

    def positioned_coords(self):
        return (self.pos @ self.coord[..., None]).squeeze()

    def dump_pdb(self, fname):
        s = ""
        for i in range(len(self.coord)):
            j = i + 1
            crd = self.positioned_coords()
            aa = self.seq[i]
            s += pdb_format_atom(resi=j, resn=aa, xyz=crd[i, 0], atomn="N")
            s += pdb_format_atom(resi=j, resn=aa, xyz=crd[i, 1], atomn="CA")
            s += pdb_format_atom(resi=j, resn=aa, xyz=crd[i, 2], atomn="C")
            s += pdb_format_atom(resi=j, resn=aa, xyz=crd[i, 3], atomn="O")
            s += pdb_format_atom(resi=j, resn=aa, xyz=crd[i, 4], atomn="CB")

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
    body2 = Body("/home/sheffler/scaffolds/C2_best/C2_3hm4_1.pdb")
    body3 = Body("/home/sheffler/scaffolds/C3_best/C3_1nza_1.pdb")

    d = body2.slide_to(body3, [1, 0, 0])

    body2.dump_pdb("body2.pdb")
    body3.dump_pdb("body3.pdb")
