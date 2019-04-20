from time import perf_counter
from sicdock.search.hierarchical import *
from sicdock.dockspec import DockSpec2CompCage
from sicdock.body import Body
from sicdock.io import dump_pdb_from_bodies


def test_2xCyclic_hier(
    C3_1nza,
    C2_3hm4,
    nfold1=3,
    nfold2=2,
    tip=0,
    ndump=0,
    base_resl=16,
    nstep=5,
    base_min_contacts=30,
    prune_frac_sortof=0.99,
    prune_minkeep=10,
    archs="I32 O32 T32",
    contact_dis=8,
):

    for arch in archs.split():
        spec = DockSpec2CompCage(arch)
        body1 = Body(C3_1nza, nfold1, which_ss="HEL")
        body2 = Body(C2_3hm4, nfold2, which_ss="HEL")

        t = perf_counter()
        npair, pos = find_connected_2xCyclic_hier_slide(
            spec,
            body1,
            body2,
            base_resl=base_resl,
            nstep=nstep,
            base_min_contacts=base_min_contacts,
            prune_frac_sortof=prune_frac_sortof,
            contact_dis=contact_dis,
            prune_minkeep=prune_minkeep,
        )
        print("search time", perf_counter() - t)
        npair0 = npair[:, 0] if npair.ndim is 2 else npair

        if len(npair0) == 0:
            print("no results")

        pos1, pos2 = spec.move_to_canonical_unit(*pos)

        # I32 grid search 1degree tip 30
        # 120*2*180*120*2 = ~10.3M ? 29K/sec?
        # search time 356.8220110750117
        # nresult 1919590
        # tcdock best 0 81
        # tcdock best 1 80
        # tcdock best 2 79
        # tcdock best 3 79
        # tcdock best 4 78
        # tcdock best 5 78
        # tcdock best 6 77
        # tcdock best 7 77
        # tcdock best 8 77
        # tcdock best 9 75

        for i, norig in enumerate(npair0):
            body1.move_to(pos1[i])
            body2.move_to(pos2[i])
            n = body1.cen_pair_count(body2, contact_dis)
            assert n == norig

        print("nresult", len(npair0))
        omax = np.argsort(-npair0)
        print(npair[omax[:10]])
        for i, imax in enumerate(omax[:ndump]):
            print("tcdock best", i, npair0[imax])
            body1.move_to(pos1[imax])
            body2.move_to(pos2[imax])
            dump_pdb_from_bodies(
                "sicdock_hier_%s_%03i.pdb" % (arch, i),
                [body1, body2],
                spec.symframes(),  # [:8],
                keep=lambda x: np.sum(x) > 0,
                no_duplicate_chains=False,
                no_duplicate_reschain_pairs=True,
                include_cen=False,
                chain_letters=-1,
            )


if __name__ == "__main__":
    import sicdock.rosetta as ros

    pose1 = ros.get_pose_cached("sicdock/data/pdb/C3_1nza_1.pdb.gz")
    pose2 = ros.get_pose_cached("sicdock/data/pdb/C2_3hm4_1.pdb.gz")

    test_2xCyclic_hier(
        pose1,
        pose2,
        ndump=3,
        archs="I32",
        tip=16,
        base_resl=16,
        nstep=5,
        base_min_contacts=0,
        prune_frac_sortof=0.875,
        contact_dis=8,
    )
