import os, sys, time, _pickle
from cppimport import import_hook
import numpy as np
from sicdock.motif.pairdat import ResPairData
from sicdock import xbin
from sicdock.rotamer import get_rotamer_space, assign_rotamers, check_rotamer_deviation
from sicdock.motif import _motif as cpp
from sicdock.motif.pairscore import ResPairScore
import getpy as gp


def bb_stubs(n, ca=None, c=None):
    if ca is None:
        assert n.ndim == 3
        assert n.shape[1] >= 3  # n, ca, c
        ca = n[:, 1, :3]
        c = n[:, 2, :3]
        n = n[:, 0, :3]

    assert len(n) == len(ca) == len(c)
    assert n.ndim == ca.ndim == c.ndim == 2
    stub = np.zeros((len(n), 4, 4))
    stub[:, 3, 3] = 1
    e1 = n[:, :3] - ca[:, :3]
    e1 /= np.linalg.norm(e1, axis=1)[:, None]
    e3 = np.cross(e1, c[:, :3] - ca[:, :3])
    e3 /= np.linalg.norm(e3, axis=1)[:, None]
    e2 = np.cross(e3, e1)
    stub[:, :3, 0] = e1
    stub[:, :3, 1] = e2
    stub[:, :3, 2] = e3
    # magic numbers from rosetta centroids in some set of pdbs
    avg_centroid_offset = [-0.80571551, -1.60735769, 1.46276045]
    t = stub[:, :3, :3] @ avg_centroid_offset + ca[:, :3]
    stub[:, :3, 3] = t
    assert np.allclose(np.linalg.det(stub), 1)
    return stub


def pair_key_ss(resi, ssi, resj, ssj, stub, cart_resl, ori_resl):
    kij = xbin.key_of_pairs2(resi, resj, stub, stub, cart_resl, ori_resl)
    kji = xbin.key_of_pairs2(resj, resi, stub, stub, cart_resl, ori_resl)
    assert kij.dtype == np.uint64
    ssi = ssi.astype("u8")
    ssj = ssj.astype("u8")
    kijss = np.bitwise_or(np.left_shift(ssi, 62), np.left_shift(ssj, 60))
    kjiss = np.bitwise_or(np.left_shift(ssj, 62), np.left_shift(ssi, 60))
    kssij = np.bitwise_or(kij, kijss)
    kssji = np.bitwise_or(kji, kjiss)
    # assert np.all(np.right_shift(kssij, 62) == ssi)
    # assert np.all(np.right_shift(kssji, 62) == ssj)
    # assert np.all(np.right_shift(np.left_shift(kssij, 2), 62) == ssj)
    # assert np.all(np.right_shift(np.left_shift(kssji, 2), 62) == ssi)
    # assert np.all(np.right_shift(np.left_shift(kssij, 4), 4) == kij)
    # assert np.all(np.right_shift(np.left_shift(kssji, 4), 4) == kji)
    return kssij, kssji


def get_pair_keys(rp, min_ssep=10, cart_resl=1, ori_resl=20):
    mask = (rp.p_resj - rp.p_resi).data >= min_ssep
    resi = rp.p_resi.data[mask]
    resj = rp.p_resj.data[mask]
    ssi = rp.ssid[resi]
    ssj = rp.ssid[resj]
    stub = rp.stub.data
    kij = np.zeros(len(rp.p_resi), dtype="u8")
    kji = np.zeros(len(rp.p_resi), dtype="u8")
    kij[mask], kji[mask] = pair_key_ss(resi, ssi, resj, ssj, stub, cart_resl, ori_resl)
    return kij, kji


def pair_key_ss_cpp(resi, resj, ss, stub, cart_resl, ori_resl):
    assert resi.dtype == ss.dtype
    kij = xbin.key_of_pairs2_ss(resi, resj, ss, ss, stub, stub, cart_resl, ori_resl)
    kji = xbin.key_of_pairs2_ss(resj, resi, ss, ss, stub, stub, cart_resl, ori_resl)
    # assert np.all(np.right_shift(kij, 62) == ss[resi])
    # assert np.all(np.right_shift(kji, 62) == ss[resj])
    # assert np.all(np.right_shift(np.left_shift(kij, 2), 62) == ss[resj])
    # assert np.all(np.right_shift(np.left_shift(kji, 2), 62) == ss[resi])
    return kij, kji


def get_pair_keys_cpp(rp, min_ssep=10, cart_resl=1, ori_resl=20):
    mask = (rp.p_resj - rp.p_resi).data >= min_ssep
    resi = rp.p_resi.data[mask]
    resj = rp.p_resj.data[mask]
    ss = rp.ssid.data
    assert np.max(ss) == 2
    assert np.min(ss) == 0
    stub = rp.stub.data
    kij = np.zeros(len(rp.p_resi), dtype="u8")
    kji = np.zeros(len(rp.p_resi), dtype="u8")
    kij[mask], kji[mask] = pair_key_ss_cpp(resi, resj, ss, stub, cart_resl, ori_resl)
    return kij, kji


def create_res_pair_score(rp, path, min_ssep=10, maxsize=None):
    N = maxsize
    keys0 = np.concatenate([rp.kij.data[:N], rp.kji.data[:N]])
    order, binkey, binrange = cpp.jagged_bin(keys0)
    assert len(binkey) == len(binrange)
    lb = np.right_shift(binrange, 32)
    ub = np.right_shift(np.left_shift(binrange, 32), 32)
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
    rps = ResPairScore(pair_score, pair_range, res1, res2, rp)
    rps.dump(path)
    return rps


def build_motif_table(rp, cart_resl=1, ori_resl=20):
    # kij, kji = get_pair_keys_cpp(rp, min_ssep=10)
    # print(kij.shape)
    # rp.data["kij"] = ["pairid"], kij
    # rp.data["kji"] = ["pairid"], kji
    # rp.attrs["xbin_type"] = "wtihss"
    # # del rp.attrs["xbin_params"]
    # # del rp.attrs["xbin_types"]
    # # del rp.attrs["xbin_swap_type"]
    # # print(rp)
    # with open(
    #     "/home/sheffler/debug/sicdock/datafiles/respairdat_si20_rotamers.pickle", "wb"
    # ) as out:
    #     _pickle.dump(rp.data, out)

    create_res_pair_score(
        rp, "/home/sheffler/debug/sicdock/datafiles/motif_score", maxsize=None
    )
    rps = ResPairScore("/home/sheffler/debug/sicdock/datafiles/motif_score")
    # print(rps)
    # print(rp)
    # rotspace = get_rotamer_space()
    # rotids, rotlbl, rotchi = assign_rotamers(rp, rotspace)
    # rp.data["rotid"] = ["resid"], rotids
    # rp.data.attrs["rotlbl"] = rotlbl
    # rp.data.attrs["rotchi"] = rotchi
    # check_rotamer_deviation(rp, rotspace)


if __name__ == "__main__":
    from sicdock.motif._loadhack import respairdat

    build_motif_table(respairdat)

    # build_motif_table(None)
