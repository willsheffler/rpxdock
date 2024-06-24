import pytest
try:
    from rpxdock.rosetta.rosetta_util import *
    from rpxdock.rosetta.helix_trix import *
    from rpxdock.rosetta.triggers_init import core, protocols, rosetta
except ImportError:
    pass
import rpxdock as rp

def test_append_helix(C2_3hm4, helix):
    pytest.importorskip('pyrosetta')
    pose = rosetta.core.pose.Pose().assign(C2_3hm4)
    og_pose = rosetta.core.pose.Pose().assign(pose)
    rp.rosetta.helix_trix.append_Nhelix(pose)
    rp.rosetta.helix_trix.append_Chelix(pose)

    assert pose.size() == og_pose.size() + 2 * helix.size()
    assert pose.sequence() == (og_pose.sequence() + 2 * helix.sequence())
    assert pose.num_chains() == 3 and og_pose.num_chains() == 1
    assert pose.chain(1) == og_pose.chain(1)

def test_termini_direction(poses_list_helices):
    pytest.importorskip('pyrosetta')
    actual_dirs = [[True, True], [False, False], [True, False]]  #based on input; change in input is changed!
    # print(poses_list_helices)
    for i in range(len(poses_list_helices)):
        pose = rosetta.core.pose.Pose().assign(poses_list_helices[i])
        rp.rosetta.helix_trix.append_Nhelix(pose)
        rp.rosetta.helix_trix.append_Chelix(pose)
        Cin = rp.rosetta.helix_trix.point_in(pose, "C", helixch=3)
        Nin = rp.rosetta.helix_trix.point_in(pose, "N", helixch=2)
        assert [Nin, Cin] == actual_dirs[i]

def test_remove_helix(C2_3hm4, helix):
    pytest.importorskip('pyrosetta')
    pose = rosetta.core.pose.Pose().assign(C2_3hm4)
    print(pose.size())
    og_pose = rosetta.core.pose.Pose().assign(pose)
    rp.rosetta.helix_trix.append_Nhelix(pose)
    rp.rosetta.helix_trix.append_Chelix(pose)
    print(pose.size())

    rp.rosetta.helix_trix.remove_helix_chain(pose, 3)
    assert pose.size() == og_pose.size() + helix.size()
    assert pose.sequence() == (og_pose.sequence() + helix.sequence())
    assert pose.num_chains() == 2 and og_pose.num_chains() == 1
    assert pose.chain(1) == og_pose.chain(1)

    rp.rosetta.helix_trix.remove_helix_chain(pose, 2)
    assert pose.size() == og_pose.size()
    assert pose.sequence() == og_pose.sequence()
    assert pose.num_chains() == 1 and og_pose.num_chains() == 1
    assert pose.chain(1) == og_pose.chain(1)
    assert pose.__eq__(og_pose)

def test_limit_flip_update():
    pytest.importorskip('pyrosetta')
    kw = rp.app.defaults()
    kw.inputs = [["foo"], ["bar"], ["xfoo", 'xbar'], ["foobear"]]
    kw.termini_dir = [[[None, None]], [[True, False]], [[False, True], [True, False]], [[False, False]]]
    kw.force_flip = [False, False, False, False]
    kw.flip_components = [True, True, True, False]
    assert limit_flip_update_pose(True, True, 1, **kw)[0]

    assert not (limit_flip_update_pose(True, True, 2, **kw)[0])
    assert limit_flip_update_pose(False, True, 2, **kw)[0]

    assert limit_flip_update_pose(False, True, 3, 0, **kw)
    assert not (limit_flip_update_pose(False, True, 3, 1, **kw)[0])

    assert not (limit_flip_update_pose(True, True, 4, **kw)[0])

def test_init_termini(mixed_inputs, helix):
    pytest.importorskip('pyrosetta')
    pose_list = mixed_inputs
    kw = rp.app.defaults()
    kw.inputs = [[inp] for inp in pose_list]
    assert len(kw.inputs) == len(pose_list) and len(kw.inputs[0]) == 1

    # Based on pose_list; change if pose_list changes
    kw.termini_dir = [[[None, None]], [[True, True]], [[False, False]], [[True, False]], [[None, None]],
                      [[True, False]]]
    kw.term_access = [[[False, False]], [[False, False]], [[True, False]], [[False, True]], [[True, True]],
                      [[False, False]]]
    kw.force_flip = [False] * len(kw.inputs)
    kw.flip_components = [True] * len(kw.inputs)

    poses, og_lens = rp.rosetta.helix_trix.init_termini(**kw)
    assert len(poses) == len(kw.inputs) == len(og_lens)
    assert og_lens == [[None], [poses[1][0].size()], [poses[2][0].size() - helix.size()],
                       [poses[3][0].size() - helix.size()], [poses[4][0].size() - (helix.size() * 2)],
                       [poses[4][0].size() - (helix.size() * 2)]]
    assert (type(poses[1][0]) == type(poses[2][0]) == type(poses[3][0]) == type(poses[4][0]) ==
            rosetta.core.pose.Pose)
    assert type(poses[0][0]) == type(poses[5][0]) == str

# @pytest.mark.skip(reason="makes body pickle for tests in test_body; not necessary to test")
# def temp_make_bodies(c3, c2):
#    import _pickle
#    from rpxdock.body import Body
#    both_c3 = rosetta.core.pose.Pose().assign(c3)
#    both_c2 = rosetta.core.pose.Pose().assign(c2)

#    kw = rp.app.defaults()
#    kw.inputs= [[c2], [both_c2], [c3], [both_c3]]
#    kw.term_access=[[[True, False]],[[True, True]],
#                   [[False, True]],[[True, True]]]
#    kw.termini_dir = [[[None,None]], [[None, None]],
#                   [[None, None]], [[None, None]]]
#    kw.flip_components = [True] * len(kw.inputs)
#    kw.force_flip = [False] * len(kw.inputs)
#    tmp_poses, og_lens = rp.rosetta.helix_trix.init_termini(**kw)

#    poses = {"C2_REFS10_1_Nhelix": c2, "C2_REFS10_1_NChelix": both_c2,
#             "C3_1na0-1_1_Chelix": c3, "C3_1na0-1_1_NChelix": both_c3}
#    # poses = {"C2_REFS10_1_Nhelix": c2, "C2_REFS10_1_NChelix": both_c2,
#    #    "C3_3e6q_asu_Chelix": c3, "C3_3e6q_asu_NChelix": both_c3}

#    bodydir = "rpxdock/data/body/"
#    i= 0
#    for key, pose in poses.items():
#       print(kw.term_access[i][0])
#       b = Body(pose, og_seqlen=og_lens[i][0], modified_term=kw.term_access[i][0])
#       with open(bodydir+key+".pickle", "wb") as out:
#          _pickle.dump(b, out)
#       i += 1

#    # # poses = ["C2_REFS10_1_Nhelix", "C2_REFS10_1_NChelix",
#    # #          "C3_1na0-1_1_Chelix", "C3_1na0-1_1_NChelix"]
#    # poses = ["C3_3e6q_asu_Chelix", "C3_3e6q_asu_NChelix"]
#    # for p in poses:
#    #    # a = get_body(p)
#    #    # print(a.modified_term)
#    #    rp.data.get_body(p).dump_pdb(f"./temp/{p}.pdb")

if __name__ == '__main__':
    from rpxdock.rosetta.triggers_init import get_pose_cached

    helix = get_pose_cached('tiny.pdb.gz', rp.data.pdbdir)
    dhr64 = get_pose_cached('dhr64.pdb.gz', rp.data.pdbdir)
    dhr14 = get_pose_cached('DHR14.pdb.gz', rp.data.pdbdir)
    poseC2 = get_pose_cached('C2_3hm4_1.pdb.gz', rp.data.pdbdir)
    poseC3 = get_pose_cached('C3_1na0-1_1.pdb.gz', rp.data.pdbdir)
    pdbC2 = (rp.data.pdbdir + '/C2_3hm4_1.pdb.gz')
    pdbtop7 = (rp.data.pdbdir + '/top7.pdb.gz')

    # test_append_helix(poseC2, helix)
    # test_termini_direction([dhr64, dhr14, poseC3])

    # Reset this pose if it was modified in the previous functions
    # poseC2 = get_pose_cached('C2_3hm4_1.pdb.gz', rp.data.pdbdir)

    # test_remove_helix(poseC2, helix)
    # test_limit_flip_update()

    # poseC2 = get_pose_cached('C2_3hm4_1.pdb.gz', rp.data.pdbdir)
    # test_init_termini([pdbtop7,dhr64, dhr14, poseC3, pdbC2, pdbC2], helix)
    # test_init_termini(helix)

    # Everything below is to make bodies modified with terminal helices,
    # which are used in tests/search/test_onecomp and test_multicomp

    # from pyrosetta import *
    # c3_forbody=pose_from_pdb("/home/jenstanisl/test_rpx/test_appendhelix/ver_scaffolds/C3_3e6q_asu.pdb")
    # temp_make_bodies(c3_forbody, get_pose_cached("C2_REFS10_1.pdb.gz", rp.data.pdbdir))
    temp_make_bodies(poseC3, get_pose_cached("C2_REFS10_1.pdb.gz", rp.data.pdbdir))

    # import _pickle
    # from rpxdock.body import Body
    # bodydir = "rpxdock/data/body/"
    # all_new_bodies=["C3_3e6q_asu","C3_1na0-1_1", "C2_REFS10_1",
    #       "C3_3e6q_asu_Chelix","C3_1na0-1_1_Chelix", "C2_REFS10_1_Nhelix",
    #       "C3_3e6q_asu_NChelix","C3_1na0-1_1_NChelix", "C2_REFS10_1_NChelix"]
    # for b in all_new_bodies:
    #    rp.data.get_body(b).dump_pdb(f"./temp/{b}.pdb")
