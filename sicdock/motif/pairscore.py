import os, _pickle
import numpy as np
import getpy as gp
from cppimport import import_hook
import sicdock.motif._motif as cpp


def load_respairscore(path):
    assert os.path.isdir(path)
    with open(os.path.join(path, "resdata.pickle"), "rb") as inp:
        rps = _pickle.load(inp)
    rps.score_map = gp.Dict(np.uint64, np.float64)
    rps.range_map = gp.Dict(np.uint64, np.uint64)
    rps.score_map.load(os.path.join(path, "score_map.bin"))
    rps.range_map.load(os.path.join(path, "range_map.bin"))
    return rps


class ResPairScore:
    def __init__(self, keys, score_map, range_map, res1, res2, rp):
        assert np.all(score_map.__contains__(keys))
        assert np.all(range_map.__contains__(keys))
        self.keys = keys
        self.score_map = score_map
        self.range_map = range_map
        self.respair = np.stack([res1.astype("i4"), res2.astype("i4")], axis=1)
        self.aaid = rp.aaid.data.astype("u1")
        self.ssid = rp.ssid.data.astype("u1")
        self.rotid = rp.rotid.data
        self.stub = rp.stub.data.astype("f4")
        self.pdb = rp.pdb.data[rp.r_pdbid.data]
        self.resno = rp.resno.data
        self.rotchi = rp.rotchi
        self.rotlbl = rp.rotlbl
        self.id2aa = rp.id2aa.data
        self.id2ss = rp.id2ss.data

    def bin_score(self, keys):
        score = np.zeros(len(keys))
        mask = self.score_map.__contains__(keys)
        score[mask] = self.score_map[keys[mask]]
        return score

    def bin_get_all_data(self, keys):
        mask = self.range_map.__contains__(keys)
        ranges = self.range_map[keys[mask]].view("u4").reshape(-1, 2)
        out = np.empty(len(ranges), dtype="O")
        for i, r in enumerate(ranges):
            lb, ub = r
            res = self.respair[lb:ub]
            rots = self.rotid[res]
            aas = self.aaid[res]
            stubs = self.stub[res]
            pdbs = self.pdb[res[:, 0]]
            resno = self.resno[res]
            # print(aas.shape, rots.shape, stubs.shape, pdbs.shape, resno.shape)
            out[i] = pdbs, resno, aas, rots, stubs
        return out, mask

    def dump(self, path):
        if os.path.exists(path):
            assert os.path.isdir(path)
        else:
            os.mkdir(path)
        self.score_map.dump(os.path.join(path, "score_map.bin"))
        self.range_map.dump(os.path.join(path, "range_map.bin"))
        tmp = self.score_map, self.range_map
        self.score_map, self.range_map = None, None
        with open(os.path.join(path, "resdata.pickle"), "wb") as out:
            _pickle.dump(self, out)
        self.score_map, self.range_map = tmp

    def bin_respairs(self, key):
        r = self.rangemap[k]
        lb = np.right_shift(r, 32)
        ub = np.right_shift(np.left_shift(r), 32)
        return self.respair[lb:ub]


def create_res_pair_score(rp, path=None, min_ssep=10, maxsize=None):
    N = maxsize
    keys0 = np.concatenate([rp.kij.data[:N], rp.kji.data[:N]])
    order, binkey, binrange = cpp.jagged_bin(keys0)
    assert len(binkey) == len(binrange)
    epair = np.concatenate([rp.p_etot.data[:N], rp.p_etot.data[:N]])[order]
    ebin = cpp.logsum_bins(binrange, -epair)
    assert len(ebin) == len(binkey)
    mask = ebin > 0.1
    pair_score = gp.Dict(np.uint64, np.float64)
    pair_score[binkey[mask]] = ebin[mask]
    # pair_score.dump("/home/sheffler/debug/sicdock/datafiles/pair_score.bin")
    pair_range = gp.Dict(np.uint64, np.uint64)
    pair_range[binkey[mask]] = binrange[mask]
    res1 = np.concatenate([rp.p_resi.data[:N], rp.p_resj.data[:N]])[order]
    res2 = np.concatenate([rp.p_resj.data[:N], rp.p_resi.data[:N]])[order]
    rps = ResPairScore(binkey[mask], pair_score, pair_range, res1, res2, rp)
    if path is not None:
        rps.dump(path)
    return rps
