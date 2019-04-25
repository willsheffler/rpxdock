import os, sys, time, _pickle
from cppimport import import_hook
import numpy as np

from sicdock import xbin
from sicdock.rotamer import get_rotamer_space, assign_rotamers, check_rotamer_deviation
from sicdock.motif import _motif as cpp
from sicdock.motif.pairscore import ResPairScore, create_res_pair_score
from sicdock.motif.pairdat import ResPairData
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
    assert resi.dtype == ss.dtype
    kij[mask] = xbin.key_of_pairs2_ss(
        resi, resj, ss, ss, stub, stub, cart_resl, ori_resl
    )
    kji[mask] = xbin.key_of_pairs2_ss(
        resj, resi, ss, ss, stub, stub, cart_resl, ori_resl
    )
    return kij, kji


def add_rots_to_respairdat(rp, rotspace):
    rotids, rotlbl, rotchi = assign_rotamers(rp, rotspace)
    rp.data["rotid"] = ["resid"], rotids
    rp.data.attrs["rotlbl"] = rotlbl
    rp.data.attrs["rotchi"] = rotchi


def add_xbin_to_respairdat(rp, min_ssep, cart_resl, ori_resl):
    kij, kji = get_pair_keys_cpp(
        rp, min_ssep=10, cart_resl=cart_resl, ori_resl=ori_resl
    )
    rp.data["kij"] = ["pairid"], kij
    rp.data["kji"] = ["pairid"], kji
    rp.attrs["xbin_type"] = "wtihss"
    if "xbin_params" in rp.attrs:
        del rp.attrs["xbin_params"]
    if "xbin_types" in rp.attrs:
        del rp.attrs["xbin_types"]
    if "xbin_swap_type" in rp.attrs:
        del rp.attrs["xbin_swap_type"]


def make_respairdat_subsets(rp):
    keep = np.arange(len(rp.pdbid))
    np.random.shuffle(keep)
    rp10 = rp.subset_by_pdb(keep[:10])
    with open("sicdock/tests/motif/respairdat10.pickle", "wb") as out:
        _pickle.dump(rp10.data, out)
    rp100 = rp.subset_by_pdb(keep[:100])
    with open("sicdock/tests/motif/respairdat100.pickle", "wb") as out:
        _pickle.dump(rp100.data, out)
    rp1000 = rp.subset_by_pdb(keep[:1000])
    with open("sicdock/tests/motif/respairdat1000.pickle", "wb") as out:
        _pickle.dump(rp1000.data, out)


def build_motif_table(rp, cart_resl=1, ori_resl=20):
    pass

    # rps = create_res_pair_score(
    # rp, "/home/sheffler/debug/sicdock/datafiles/respairscore100", maxsize=None
    # )

    # rps = ResPairScore("/home/sheffler/debug/sicdock/datafiles/motif_score")
    # print(rps)

    # print(rp)

    # f = "/home/sheffler/debug/sicdock/datafiles/respairdat_si30_rotamers.pickle"
    # add_xbin_to_respairdat(rp, f)


if __name__ == "__main__":
    from sicdock.motif._loadhack import respairdat

    build_motif_table(respairdat)

    # build_motif_table(None)
