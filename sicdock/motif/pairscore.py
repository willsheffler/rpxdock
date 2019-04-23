import os, _pickle
import numpy as np
import getpy as gp


class ResPairScore:
    def __init__(self, score_map, range_map=None, res1=None, res2=None, rp=None):
        if isinstance(score_map, str):
            self.load(score_map)
            return
        self.score_map = score_map
        self.range_map = range_map
        self.respair = np.stack([res1.astype("i4"), res2.astype("i4")], axis=1)
        self.aaid = rp.aaid.data.astype("u1")
        self.ssid = rp.ssid.data.astype("u1")
        self.rotid = rp.rotid.data
        self.rotchi = rp.rotchi
        self.rotlbl = rp.rotlbl
        self._fields = (
            self.respair,
            self.aaid,
            self.ssid,
            self.rotid,
            self.rotchi,
            self.rotlbl,
        )

    def bin_score(self, keys):
        score = np.zeros(len(keys))
        mask = self.score_map.__contains__(keys)
        score[mask] = self.score_map[keys[mask]]
        score[~mask] = 0
        return score

    def dump(self, path):
        if os.path.exists(path):
            assert os.path.isdir(path)
        else:
            os.mkdir(path)
        with open(path + "/resdata.pickle", "wb") as out:
            _pickle.dump((self._fields), out)
        self.score_map.dump(path + "/score_map.bin")
        self.range_map.dump(path + "/range_map.bin")

    def load(self, path):
        self.score_map = gp.Dict(np.uint64, np.float64)
        self.range_map = gp.Dict(np.uint64, np.uint64)
        assert os.path.isdir(path)
        with open(path + "/resdata.pickle", "rb") as inp:
            self._fields = _pickle.load(inp)
        self.score_map.load(path + "/score_map.bin")
        self.range_map.load(path + "/range_map.bin")

    def bin_respairs(self, key):
        r = self.rangemap[k]
        lb = np.right_shift(r, 32)
        ub = np.right_shift(np.left_shift(r), 32)
        return self.respair[lb:ub]
