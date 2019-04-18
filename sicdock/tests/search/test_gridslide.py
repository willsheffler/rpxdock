import numpy as np
import homog as hm

from sicdock.body import Body
from sicdock.dockspec import DockSpec2CompCage, DockSpec1CompCage
from sicdock.search import gridslide
from sicdock.io import dump_pdb
from sicdock import sym


def test_1xCyclic(
    C3_1nza,
    C2_3hm4,
    nfold1=3,
    nfold2=2,
    ndump=0,
    resl=10,
    tip=1,
    contact_dis=8,
    min_contacts=10,
):
    # for arch in "T2".split():
    for arch in "I3 I2 O3 O2 T3 T2".split():
        nfold = int(arch[1])
        pose = C3_1nza if nfold is 3 else C2_3hm4
        spec = DockSpec1CompCage(arch)
        body = Body(pose, nfold, which_ss="HEL")

        samples = gridslide.samples_1xCyclic(spec, resl=resl)
        npair, pos = gridslide.find_connected_1xCyclic(
            spec, body, samples, min_contacts=min_contacts, contact_dis=contact_dis
        )

        body2 = body.copy()
        pos2 = spec.placements_second(pos)
        for i, norig in enumerate(npair):
            body.move_to(pos[i])
            body2.move_to(pos2[i])
            n = body.cen_pair_count(body2, contact_dis)
            assert n == norig

        omax = np.argsort(-npair)
        for i, imax in enumerate(omax[:ndump]):
            best = npair[imax]
            print("tcdock best", i, best, "nresult", len(npair))
            body.move_to(pos[imax])
            dump_pdb(
                "sicdock_%s_%03i.pdb" % (arch, i),
                body,
                spec.symframes(),  # [:8],
                keep=lambda x: x[2] >= 0,
                no_duplicate_chains=False,
                no_duplicate_reschain_pairs=True,
                include_cen=False,
                chain_letters=-1,
            )


def test_2xCyclic(
    C3_1nza,
    C2_3hm4,
    nfold1=3,
    nfold2=2,
    ndump=0,
    resl=10,
    tip=1,
    contact_dis=8,
    min_contacts=10,
):

    for arch in "I32 O32 T32".split():
        spec = DockSpec2CompCage(arch)
        body1 = Body(C3_1nza, nfold1, which_ss="HEL")
        body2 = Body(C2_3hm4, nfold2, which_ss="HEL")

        samples = gridslide.samples_2xCyclic(
            spec, resl=resl, max_out_of_plane_angle=tip
        )
        npair, pos1, pos2 = gridslide.find_connected_2xCyclic(
            spec,
            body1,
            body2,
            samples,
            min_contacts=min_contacts,
            contact_dis=contact_dis,
        )
        pos1, pos2 = spec.move_to_canonical_unit(pos1, pos2)
        if len(npair) == 0:
            print("no results")

        assert np.all(npair >= min_contacts)

        for i, norig in enumerate(npair):
            body1.move_to(pos1[i])
            body2.move_to(pos2[i])
            n = body1.cen_pair_count(body2, contact_dis)
            assert n == norig

        omax = np.argsort(-npair)
        for i, imax in enumerate(omax[:ndump]):
            best = npair[imax]
            print("tcdock best", i, best, "nresult", len(npair))
            body1.move_to(pos1[imax])
            body2.move_to(pos2[imax])
            dump_pdb(
                "sicdock_%s_%03i.pdb" % (arch, i),
                [body1, body2],
                spec.symframes(),  # [:8],
                # keep=lambda x: np.sum(x) > 0,
                no_duplicate_chains=False,
                no_duplicate_reschain_pairs=True,
                include_cen=False,
                chain_letters=-1,
            )


if __name__ == "__main__":

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

    test_1xCyclic(pose1, pose2, ndump=0, resl=10)
    test_2xCyclic(pose1, pose2, ndump=0, resl=10)
