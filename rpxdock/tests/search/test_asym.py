import sys
import pytest
import numpy as np
import rpxdock as rp
import willutil as wu
from concurrent.futures import ThreadPoolExecutor

pytest.importorskip('pyrosetta')

def main():
    # hscore = rp.data.small_hscore()
    hscore = rp.RpxHier('ilv_h/1000', hscore_data_dir='/home/sheffler/data/rpx/hscore/willsheffler')
    # hscore = rp.RpxHier('afilmv_elh/10', hscore_data_dir='/home/sheffler/data/rpx/hscore/willsheffler/')
    # hscore.score_only_sspair = ['HH']

    # body1 = rp.data.get_body('DHR14')
    # body2 = rp.data.get_body('top7')
    # body1.score_only_ss = 'H'
    # body2.score_only_ss = 'H'
    # body1 = rp.Body('rpxdock/data/pdb/DHR14.pdb.gz')
    # body2 = rp.Body('rpxdock/data/pdb/top7.pdb.gz')
    body2 = rp.Body('/home/sheffler/src/BFF/rf_diffusion/tmp.pdb')

    # body1 = rp.data.get_body('top7b')

    # test_asym(hscore, body1, body2)

    test_dockasym_symframes(hscore, body2)
    # assert 0
    # test_asym_trim(hscore, body1, body2)

    # _test_asym_2iface()
    ic('test_asym.py DONE')

def test_dockasym_symframes(hscore, body):
    kw = _test_args()
    kw.max_trim = 0
    kw.output_prefix = 'test_asym'
    kw.beam_size = 4e4
    cartlb = np.array([-30, -20, -20])
    cartub = np.array([+30, +20, +20])
    # limit_rotation = np.pi / 3
    limit_rotation = 0.5
    cartbs = np.array([6, 4, 4], dtype='i')
    sampler = rp.sampling.XformHier_f4(cartlb, cartub, cartbs, 20)
    kw.wts = wu.Bunch(ncontact=0.0, rpx=1.0)
    kw.max_bb_redundancy = 0.666
    sym = 'oct'
    ax2 = wu.sym.axes(sym)[2]
    ax3 = wu.sym.axes(sym)[3]
    frames = [np.eye(4), wu.hrot(ax2, 180), wu.hrot(ax3, 120), wu.hrot(ax3, 240)]

    # calc allowed_res1 and 2
    rsets0, rsets1 = list(), list()
    for iframe in range(len(frames)):
        if iframe in [0]: continue
        for maxdisx2 in range(5, 25):
            pairs = body.contact_pairs(body, frames[0], frames[iframe], maxdis=maxdisx2 / 2)
            required_res_set0 = set(pairs[:, 0])
            required_res_set1 = set(pairs[:, 1])
            # for r in rsets0:
            # required_res_set0 -= r
            if len(required_res_set0) >= 7 and len(required_res_set1) >= 7:
                # ic('break', maxdisx2 / 2, len(required_res_set0), len(required_res_set1))
                rsets0.append(required_res_set0)
                rsets1.append(required_res_set1)
                break
    rsets0 = [list(sorted(r)) for r in rsets0]
    rsets1 = [list(sorted(r)) for r in rsets1]
    # ic(rsets0)
    # ic(rsets1)
    # body.set_allowed_residues(required_res_sets=rsets)
    bodies0 = [body.copy() for b in frames[:-1]]
    bodies1 = [body.copy() for b in frames[1:]]
    # for i in range(len(bodies0)):
    # bodies0[i].set_allowed_residues(required_res_sets=[rsets0[i]])
    # bodies1[i].set_allowed_residues(required_res_sets=[rsets1[i]])

    # body = body.copy_xformed(wu.htrans((ax2 + ax2) * 40))
    x2asymcen = wu.htrans(-body.com())
    # body = body.copy_xformed(wu.htrans(-body.com()))

    result = rp.search.make_asym(
        [bodies0, bodies1],
        hscore,
        sampler,
        frames=frames,
        x2asymcen=x2asymcen,
        limit_rotation=limit_rotation,
        clashdist=2,
        **kw,
    )

    print(result)
    result.dump_pdbs_top_score(10, symframes=frames)

    # rp.search.result_to_tarball(result, 'rpxdock/data/testdata/test_asym.result', overwrite=True)
    ref = rp.data.get_test_data('test_asym')
    try:
        rp.search.assert_results_close(result, ref)
    except AssertionError:
        print('WARNING full results for asym docking dont match... checking scores only')
        assert np.allclose(ref.scores, result.scores, atol=1e-6)

def _test_args():
    kw = rp.app.defaults()
    kw.wts = wu.Bunch(ncontact=0.01, rpx=1.0)
    kw.beam_size = 1e4
    kw.max_bb_redundancy = 3.0
    kw.max_longaxis_dot_z = 0.5
    if 'pytest' not in sys.modules:
        kw.executor = ThreadPoolExecutor(min(4, kw.ncpu / 2))
    kw.multi_iface_summary = np.min
    kw.debug = True
    return kw

def _test_asym_2iface():
    kw = rp.app.defaults()
    kw.beam_size = 500_000
    kw.allowed_residues

    # kw.generate_hscore_pickle_files = True
    hscore = rp.RpxHier('ilv_h/1000', **kw)
    # ic('foo')
    cartlb = np.array([-150, -150, -150])
    cartub = np.array([+150, +150, +150])
    # cartbs = np.array([10, 10, 10], dtype="i")
    cartbs, oresl = np.array([10, 10, 10], dtype="i"), 35
    sampler = rp.sampling.XformHier_f4(cartlb, cartub, cartbs, oresl)

    from rpxdock.rosetta.triggers_init import get_pose_cached
    pose1 = get_pose_cached('dhr64.pdb.gz')
    pose2 = get_pose_cached('DHR14.pdb.gz')
    require1 = [np.arange(50), np.arange(pose1.size() - 50, pose1.size())]
    require2 = [np.arange(50), np.arange(pose2.size() - 50, pose2.size())]
    body1 = rp.Body(pose1, allowed_res=require1[0], required_res_sets=require1)
    body2 = rp.Body(pose2, allowed_res=require2[0], required_res_sets=require2)

    result = rp.search.make_asym([body2, body1], hscore, sampler, **kw)
    # rp.search.result_to_tarball(result, 'rpxdock/data/testdata/test_asym_2iface.result', overwrite=True)
    # ref = rp.data.get_test_data('test_asym_2iface')

    result.dump_pdbs_top_score(8)
    assert 0

def test_asym(hscore, body, body2):
    kw = _test_args()
    kw.max_trim = 0
    kw.output_prefix = 'test_asym'
    cartlb = np.array([+00, +10, +00])
    cartub = np.array([+30, +20, +30])
    cartbs = np.array([4, 1, 4], dtype="i")
    sampler = rp.sampling.XformHier_f4(cartlb, cartub, cartbs, 30)
    result = rp.search.make_asym([body2, body], hscore, sampler, **kw)

    # result.dump_pdbs_top_score(3)

    # rp.search.result_to_tarball(result, 'rpxdock/data/testdata/test_asym.result', overwrite=True)
    ref = rp.data.get_test_data('test_asym')

    try:
        rp.search.assert_results_close(result, ref)
    except AssertionError:
        print('WARNING full results for asym docking dont match... checking scores only')
        assert np.allclose(ref.scores, result.scores, atol=1e-6)

@pytest.mark.skip
def test_asym_trim(hscore, body, body2):
    kw = _test_args()
    kw.max_trim = 100
    kw.output_prefix = 'test_asym_trim'

    kw.beam_size = 2e5
    kw.executor = None

    cartlb = np.array([-40, +00, -40])
    cartub = np.array([+40, +40, +40])
    cartbs = np.array([12, 12, 12], dtype="i")
    sampler = rp.sampling.XformHier_f4(cartlb, cartub, cartbs, 30)

    # sampler = rp.search.asym_get_sample_hierarchy(body2, hscore, 18)
    # print(f'toplevel samples {sampler.size(0):,}')
    result = rp.search.make_asym([body2, body], hscore, sampler, **kw)

    # rp.dump(result, 'rpxdock/data/testdata/test_asym_trim.pickle')
    ref = rp.data.get_test_data('test_asym_trim')
    rp.search.assert_results_close(result, ref)

if __name__ == '__main__':
    main()
