import time
from sicdock.motif import *


def test_create_res_pair_score(respairdat, tmpdir):
    rps = create_res_pair_score(respairdat, min_ssep=10)
    rps.dump(str(tmpdir))
    rps2 = load_respairscore(str(tmpdir))
    assert np.all(rps.rotid == rps2.rotid)
    assert np.all(rps.pdb == rps2.pdb)


def test_pair_score(respairscore):
    rps = respairscore
    print(rps.respair.shape, rps.ssid.shape)

    keys = rps.keys
    scores = rps.score_map[keys]
    ranges = rps.range_map[keys].view("u4").reshape(-1, 2)
    nrange = ranges[:, 1] - ranges[:, 0]
    rlb = ranges[:, 0]
    rub = ranges[:, 1]

    ssi = rps.ssid[rps.respair[:, 0]]
    ssj = rps.ssid[rps.respair[:, 1]]

    for ibin in range(len(keys)):
        # print(ibin, scores[ibin], nrange[ibin], ranges[ibin])
        lb = rlb[ibin]
        ub = rub[ibin]
        assert len(np.unique(ssi[lb:ub])) == 1
        assert len(np.unique(ssj[lb:ub])) == 1


def test_bin_score(respairscore):
    keys = np.zeros(len(respairscore.keys) * 2, dtype="u8")
    keys[: len(respairscore.keys)] = respairscore.keys
    np.random.shuffle(keys)
    t = time.perf_counter()
    s = respairscore.bin_score(keys)
    t = time.perf_counter() - t
    print(f"perf {int(len(s) / t):,} {len(s):,}")


def test_bin_get_all_data(respairscore):
    # on real system: perf perrot: 434,667 perkey: 141,453

    assert not respairscore.range_map.__contains__(0)
    keys = np.zeros(len(respairscore.keys) * 2, dtype="u8")
    keys[: len(respairscore.keys)] = respairscore.keys
    np.random.shuffle(keys)

    t = time.perf_counter()
    binrots, hits = respairscore.bin_get_all_data(keys)
    t = time.perf_counter() - t
    print(len(keys), len(binrots))
    for i in range(10):
        print([x.shape for x in binrots[i]])

    assert np.all(keys[hits] > 0)
    assert np.all(keys[~hits] == 0)

    totsize = sum(len(x[0]) for x in binrots)
    print(len(binrots), totsize)
    print(f"perf perrot: {int(totsize / t):,} perkey: {int(len(binrots) / t):,}")


if __name__ == "__main__":
    import _pickle

    # f = "/home/sheffler/debug/sicdock/datafiles/respairdat_si30_rotamers.pickle"
    # f2 = "/home/sheffler/debug/sicdock/datafiles/respairdat_si30_rotamers"
    # f = "sicdock/data/respairdat10.pickle"
    f2 = "sicdock/data/respairscore10"
    # with open(f, "rb") as inp:
    #     rp = ResPairData(_pickle.load(inp))
    #     test_create_res_pair_score(rp, f2)

    rps = load_respairscore(f2)
    # test_pair_score(rps)
    test_bin_get_all_data(rps)
    # test_bin_score(rps)
