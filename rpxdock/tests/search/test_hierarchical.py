# from time import perf_counter
# from rpxdock.search.hierarchical import *
# from rpxdock.search.dockspec import DockSpec2CompCage
# from rpxdock.body import Body
# from rpxdock.io.io_body import dump_pdb_from_bodies
# from rpxdock.cluster import prune_results_2comp

# def test_tccage_slide_hier_depricated(
#       C3_1nza,
#       C2_3hm4,
#       nfold1=3,
#       nfold2=2,
#       tip=0,
#       ndump=0,
#       base_resl=16,
#       nstep=5,
#       base_min_contacts=30,
#       prune_frac_sortof=0.99,
#       prune_minkeep=10,
#       archs="I32 O32 T32",
#       contact_dis=8,
# ):
#
#    for arch in archs.split():
#       spec = DockSpec2CompCage(arch)
#       body1 = Body(C3_1nza, sym=nfold1, score_only_ss="HEL")
#       body2 = Body(C2_3hm4, sym=nfold2, score_only_ss="HEL")
#
#       t = perf_counter()
#       npair, pos = tccage_slide_hier_depricated(
#          spec,
#          body1,
#          body2,
#          base_resl=base_resl,
#          nstep=nstep,
#          base_min_contacts=base_min_contacts,
#          prune_frac_sortof=prune_frac_sortof,
#          contact_dis=contact_dis,
#          prune_minkeep=prune_minkeep,
#       )
#       print("search time", perf_counter() - t)
#       npair0 = npair[:, 0] if npair.ndim is 2 else npair
#
#       if len(npair0) == 0:
#          print("no results")
#
#       prev = len(npair0), npair0[np.argsort(-npair0)[:20]]
#       npair0, pos = prune_results_2comp(spec, body1, body1, npair0, pos, 5)
#       print(prev[0], len(npair0), len(npair0) / prev[0])
#       print(prev[1])
#       print(npair0[:20])
#
#       # assert 0
#
#       pos1, pos2 = spec.move_to_canonical_unit(*pos)
#
#       for i, norig in enumerate(npair0):
#          body1.move_to(pos1[i])
#          body2.move_to(pos2[i])
#          n = body1.contact_count(body2, contact_dis)
#          assert n == norig
#
#       omax = np.argsort(-npair0)
#       for i, imax in enumerate(omax[:ndump]):
#          body1.move_to(pos1[imax])
#          body2.move_to(pos2[imax])
#          dump_pdb_from_bodies(
#             "rpxdock_hier_%s_%03i.pdb" % (arch, i),
#             [body1, body2],
#             spec.symframes(),  # [:8],
#             keep=lambda x: np.sum(x) > 0,
#             no_duplicate_chains=False,
#             no_duplicate_reschain_pairs=True,
#             include_cen=False,
#             chain_letters=-1,
#          )

# if __name__ == "__main__":
#    from rpxdock.rosetta.triggers_init import get_pose_cached
#
#    pose1 = get_pose_cached("rpxdock/data/pdb/C3_1nza_1.pdb.gz")
#    pose2 = get_pose_cached("rpxdock/data/pdb/C2_3hm4_1.pdb.gz")
#
#    test_tccage_slide_hier_depricated(
#       pose1,
#       pose2,
#       ndump=10,
#       archs="O32",
#       tip=32,
#       base_resl=16,
#       nstep=5,
#       base_min_contacts=0,
#       prune_frac_sortof=0.85,
#       prune_minkeep=100,
#       contact_dis=8,
#    )
#
# test_tccage_slide_hier_depricated(
#     pose1,
#     pose2,
#     ndump=10,
#     archs="O32",
#     tip=16,
#     base_resl=16,
#     nstep=5,
#     base_min_contacts=0,
#     prune_frac_sortof=0.875,
#     contact_dis=8,
# )

# full grid search
# tcdock best 0 81
# tcdock best 1 80
# tcdock best 2 80
# tcdock best 3 79
# tcdock best 4 79
# tcdock best 5 78
# tcdock best 6 77
# tcdock best 7 76
# tcdock best 8 76
# tcdock best 9 75

# search time 4.884451340010855
# nresult 1159
#  81 81
#  80 80
#  80 80
#  79 79
#  79 79
#  78 78
#  77 77
#  76 76
#  76 76
#  75 75
