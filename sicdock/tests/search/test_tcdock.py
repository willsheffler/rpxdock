import numpy as np
import homog as hm

from sicdock.body import Body
from sicdock.search import Architecture, get_connected_architectures
from sicdock.io import dump_pdb


def test_tcdock(C3_1nza, C2_3hm4, sym1=3, sym2=2):

    body1 = Body(C3_1nza, sym1)
    body2 = Body(C2_3hm4, sym2)

    arch = Architecture("T32")
    npair, pos1, pos2 = get_connected_architectures(body1, body2, arch, resl=10)

    amin = np.argmax(npair)
    best = npair[amin]
    bestpos1 = pos1[amin]
    bestpos2 = pos2[amin]
    print("tcdock best", best, "nresult", len(npair))
    print(bestpos1)
    print(bestpos2)

    body1.move_to(bestpos1)
    body2.move_to(bestpos2)

    # dump_pdb(
    # "assembly.pdb", [body1, body2], arch.symframes(), keep=lambda x: np.sum(x) > 0
    # )


if __name__ == "__main__":

    # arch = Architecture("T32")
    # ofst = hm.htrans(arch.slide_dir(0) * 100)
    # arch.place_bodies(arch.orig1 + ofst, arch.orig2)
    # assert 0
    import sicdock.rosetta as ros

    f1 = "sicdock/data/pdb/C3_1nza_1.pdb.gz"
    f2 = "sicdock/data/pdb/C2_3hm4_1.pdb.gz"
    # f1 = "/home/sheffler/scaffolds/big/C2_3jpz_1.pdb"
    # f2 = "/home/sheffler/scaffolds/big/C3_3ziy_1.pdb"
    # f1 = "/home/sheffler/scaffolds/wheel/C3.pdb"
    # f2 = "/home/sheffler/scaffolds/wheel/C5.pdb"
    pose1 = ros.get_pose_cached(f1)
    pose2 = ros.get_pose_cached(f2)
    # pose1 = ros.pose_from_file(f1)
    # pose2 = ros.pose_from_file(f2)

    test_tcdock(pose1, pose2)
