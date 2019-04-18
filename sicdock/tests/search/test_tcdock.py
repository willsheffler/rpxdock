import numpy as np
import homog as hm

from sicdock.body import Body
from sicdock.search import tcdock, Arch


def test_tcdock(C3_1nza, C2_3hm4, sym1=3, sym2=2):

    body1 = Body(C3_1nza, sym1)
    body2 = Body(C2_3hm4, sym2)

    arch = Arch("T32")
    resl = 10
    best, bestpos = tcdock(body1, body2, arch, resl=5)
    print("tcdock best", best)
    print(bestpos)

    oldpos1, oldpos2 = bestpos
    bestpos = arch.place_bodies(*bestpos)
    body1.move_to(bestpos[0])
    body2.move_to(bestpos[1])
    body1.dump_pdb("body1_asym.pdb", asym=True)
    body2.dump_pdb("body2_asym.pdb", asym=True)

    xalign = bestpos[0] @ np.linalg.inv(oldpos1)
    assert np.allclose(bestpos[1], xalign @ oldpos2)

    # body1.move_to(xalign @ oldpos1)
    # body2.move_to(xalign @ oldpos2)
    # body1.dump_pdb("body1_sym.pdb")
    # body2.dump_pdb("body2_sym.pdb")


if __name__ == "__main__":

    # arch = Arch("T32")
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
