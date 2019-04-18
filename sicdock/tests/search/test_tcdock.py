import numpy as np
import homog as hm

from sicdock.body import Body
from sicdock.search import (
    Architecture,
    get_connected_architectures,
    get_cyclic_cyclic_samples,
)
from sicdock.io import dump_pdb


def test_tcdock(C3_1nza, C2_3hm4, sym1=3, sym2=2):

    arch = Architecture("I32")
    body1 = Body(C3_1nza, sym1, which_ss="HE")
    body2 = Body(C2_3hm4, sym2, which_ss="HE")

    samples = get_cyclic_cyclic_samples(arch, resl=10, max_out_of_plane_angle=1)
    npair, pos1, pos2 = get_connected_architectures(
        arch, body1, body2, samples, min_contacts=10
    )
    pos1, pos2 = arch.move_to_canonical_unit(pos1, pos2)
    if len(npair) == 0:
        print("no results")
    omax = np.argsort(-npair)
    for i, imax in enumerate(omax[:10]):
        best = npair[imax]
        bestpos1 = pos1[imax]
        bestpos2 = pos2[imax]
        print("tcdock best", i, best, "nresult", len(npair))
        body1.move_to(bestpos1)
        body2.move_to(bestpos2)
        dump_pdb(
            "sicdock%03i.pdb" % i,
            [body1, body2],
            arch.symframes()[:8],
            # keep=lambda x: np.sum(x) > 0,
            no_duplicate_chains=False,
        )


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
