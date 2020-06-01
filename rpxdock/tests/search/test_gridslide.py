# from time import perf_counter
# import numpy as np
# import rpxdock.homog as hm
#
# from rpxdock.search.hierarchical import tccage_slide_hier_samples_depricated
# from rpxdock.body import Body
# from rpxdock.search.dockspec import (
#    DockSpec2CompCage,
#    DockSpec1CompCage,
#    DockSpecMonomerToCyclic,
# )
# from rpxdock.search import gridslide
# from rpxdock.io.io_body import dump_pdb_from_bodies
# from rpxdock.geom import sym
#
# from rpxdock.cluster.prune import prune_results_2comp
#
# def test_1xCyclic(
#       C3_1nza,
#       C2_3hm4,
#       nfold1=3,
#       nfold2=2,
#       ndump=0,
#       resl=0.9,
#       contact_dis=8,
#       min_contacts=10,
# ):
#    # for arch in "T2".split():
#    for arch in "I3 I2 O3 O2 T3 T2".split():
#       nfold = int(arch[1])
#       pose = C3_1nza if nfold is 3 else C2_3hm4
#       spec = DockSpec1CompCage(arch)
#       body = Body(pose, nfold, score_only_ss="HEL")
#
#       samples = gridslide.samples_1xCyclic(spec, resl=resl)
#       t = perf_counter()
#       npair, pos = gridslide.find_connected_1xCyclic_slide(
#          spec, body, samples, min_contacts=min_contacts, contact_dis=contact_dis)
#       print("search time", perf_counter() - t)
#
#       body2 = body.copy()
#       pos2 = spec.placements_second(pos)
#       for i, norig in enumerate(npair):
#          body.move_to(pos[i])
#          body2.move_to(pos2[i])plug
#          n = body.contact_count(body2, contact_dis)
#          assert n == norig
#
#       omax = np.argsort(-npair)
#       for i, imax in enumerate(omax[:ndump]):
#          print("tcdock best", i, npair[imax], "nresult", len(npair))
#          body.move_to(pos[imax])
#          # dump_pdb_from_bodies(
#          #    "rpxdock_%s_%03i.pdb" % (arch, i),
#          #    body,
#          #    spec.symframes(),  # [:8],
#          #    keep=lambda x: x[2] >= 0,
#          #    no_duplicate_chains=False,
#          #    no_duplicate_reschain_pairs=True,
#          #    include_cen=False,
#          #    chain_letters=-1,
#          # )
#
# def test_2xCyclic(
#       C3_1nza,
#       C2_3hm4,
#       nfold1=3,
#       nfold2=2,
#       ndump=0,
#       resl=12,
#       tip=12,
#       contact_dis=8,
#       min_contacts=10,
#       archs="I32 O32 T32",
#       use_hier_leaf_samples=False,
# ):
#
#    for arch in archs.split():
#       spec = DockSpec2CompCage(arch)
#       body1 = Body(C3_1nza, nfold1, score_only_ss="HEL")
#       body2 = Body(C2_3hm4, nfold2, score_only_ss="HEL")
#
#       if use_hier_leaf_samples:
#          samples, *_ = tccage_slide_hier_samples_depricated(spec, resl=resl,
#                                                             max_out_of_plane_angle=tip, nstep=5)
#       else:
#          samples = gridslide.samples_2xCyclic_slide(spec, resl=resl, max_out_of_plane_angle=tip)
#       print("samples tip:", tip, len(samples[0]), len(samples[1]), len(samples[2]))
#
#       t = perf_counter()
#       npair, pos = gridslide.find_connected_2xCyclic_slide(
#          spec,
#          body1,
#          body2,
#          samples,
#          min_contacts=min_contacts,
#          contact_dis=contact_dis,
#          onebody=0,
#       )
#       print("search time", perf_counter() - t)
#       if npair.ndim > 1:
#          npair = npair[0]
#
#       prev = len(npair), npair[np.argsort(-npair)[:20]]
#       npair, pos = prune_results_2comp(spec, body1, body1, npair, pos, 5)
#       print(len(npair) / prev[0])
#       print(prev[1])
#       print(npair[:20])
#
#       pos1, pos2 = spec.move_to_canonical_unit(*pos)
#       if len(npair) == 0:
#          print("no results")
#
#       assert np.all(npair >= min_contacts)
#
#       for i, norig in enumerate(npair):
#          body1.move_to(pos1[i])
#          body2.move_to(pos2[i])
#          n = body1.contact_count(body2, contact_dis)
#          assert n == norig
#
#       omax = np.argsort(-npair)  # order by 1b contact
#       print("nresult", len(npair))
#       for i, imax in enumerate(omax[:ndump]):
#          print("tcdock best", i, npair[imax])
#          body1.move_to(pos1[imax])
#          body2.move_to(pos2[imax])
#          # dump_pdb_from_bodies(
#          #    "rpxdock_%s_%03i.pdb" % (arch, i),
#          #    [body1, body2],
#          #    spec.symframes(),  # [:8],
#          #    keep=lambda x: np.sum(x) > 0,
#          #    no_duplicate_chains=False,
#          #    no_duplicate_reschain_pairs=True,
#          #    include_cen=False,
#          #    chain_letters=-1,
#          # )
#
# def test_monomer_to_cyclic(top7, ndump=0, resl=30, contact_dis=8, min_contacts=10,
#                            archs="C2 C3 C4 C5 C6"):
#    for arch in archs.split():
#       spec = DockSpecMonomerToCyclic(arch)
#       body = Body(top7, 1, score_only_ss="HEL")
#       samples = gridslide.samples_1xMonomer_orientations(resl)
#       npair, pos = gridslide.find_connected_monomer_to_cyclic_slide(
#          spec, body, samples, min_contacts=min_contacts, contact_dis=contact_dis)
#
#       body2 = body.copy()
#       pos2 = spec.placements_second(pos)
#       for i, norig in enumerate(npair):
#          body.move_to(pos[i])
#          body2.move_to(pos2[i])
#          n = body.contact_count(body2, contact_dis)
#          assert n == norig
#
#       omax = np.argsort(-npair)
#       for i, imax in enumerate(omax[:ndump]):
#          best = npair[imax]
#          print("tcdock best", i, best, "nresult", len(npair))
#          body.move_to(pos[imax])
#          # dump_pdb_from_bodies(
#          #    "rpxdock_%s_%03i.pdb" % (arch, i),
#          #    body,
#          #    spec.symframes(),  # [:8],
#          #    keep=lambda x: x[2] >= 0,
#          #    no_duplicate_chains=False,
#          #    no_duplicate_reschain_pairs=True,
#          #    include_cen=False,
#          #    chain_letters=-1,
#          # )
#
# if __name__ == "__main__":
#
#    from rpxdock.rosetta.triggers_init import get_pose_cached
#
#    f1 = "rpxdock/data/pdb/C3_1nza_1.pdb.gz"
#    f2 = "rpxdock/data/pdb/C2_3hm4_1.pdb.gz"
#    # f1 = "/home/sheffler/scaffolds/big/C2_3jpz_1.pdb"
#    # f2 = "/home/sheffler/scaffolds/big/C3_3ziy_1.pdb"
#    # f1 = "/home/sheffler/scaffolds/wheel/C3.pdb"
#    # f2 = "/home/sheffler/scaffolds/wheel/C5.pdb"
#    pose1 = get_pose_cached(f1)
#    pose2 = get_pose_cached(f2)
#    # test_1xCyclic(pose1, pose2, ndump=0, resl=1)
#    # test_2xCyclic(pose1, pose2, ndump=3, resl=10, archs="T32")
#
#    test_2xCyclic(pose1, pose2, nfold1=3, nfold2=2, ndump=10, resl=2, tip=0, archs="I32")
#    # full run
#    # test_2xCyclic(
#    #     pose1,
#    #     pose2,
#    #     nfold1=3,
#    #     nfold2=2,
#    #     ndump=10,
#    #     resl=16,
#    #     tip=16,
#    #     archs="O32",
#    #     use_hier_leaf_samples=True,
#    # )
#
#    # pose3 = get_pose_cached("rpxdock/data/pdb/C5_1ojx.pdb.gz")
#    # test_2xCyclic(
#    # pose3, pose1, nfold1=5, nfold2=3, ndump=3, resl=5, tip=10, archs="I53"
#    # )
#
#    # top7 = get_pose_cached("rpxdock/data/pdb/top7.pdb.gz")
#    # test_monomer_to_cyclic(top7, ndump=10, resl=3, archs="C4")
