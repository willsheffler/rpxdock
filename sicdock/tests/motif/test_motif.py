from sicdock.motif import *
import pytest


def test_jagged_bin():
    kij = np.random.randint(1, 9e2, int(9e3)).astype("u8")
    kji = np.random.randint(1, 9e2, int(9e3)).astype("u8")
    k = np.concatenate([kij, kji])
    order, binkey, binrange = cpp.jagged_bin(k)
    kord = k[order]
    assert np.all(np.diff(kord) >= 0)
    lb, ub = binrange.view("u4").reshape(-1, 2).swapaxes(0, 1)
    assert np.sum(ub == lb) == 0, "not empty"
    assert np.all(kord[lb] == binkey)
    assert np.all(kord[ub - 1] == binkey)


def test_jagged_bin_zero():
    N = 1000
    kij = np.random.randint(1, 9e2, N).astype("u8")
    kji = np.random.randint(1, 9e2, N).astype("u8")
    kij[np.random.randint(0, 2, N) == 0] = 0
    kji[np.random.randint(0, 2, N) == 0] = 0
    k = np.concatenate([kij, kji])
    order, binkey, binrange = cpp.jagged_bin(k)
    kord = k[order]
    assert np.all(np.diff(kord) >= 0)
    lb, ub = binrange.view("u4").reshape(-1, 2).swapaxes(0, 1)
    assert np.sum(ub == lb) == 0, "not empty"
    assert np.all(kord[lb] == binkey)
    assert np.all(kord[ub - 1] == binkey)


def test_pair_key(respairdat):
    rp = respairdat
    kij, kji = get_pair_keys_cpp(rp, min_ssep=0)
    kij2, kji2 = get_pair_keys(rp, min_ssep=0)
    assert np.all(kij == kij2)
    assert np.all(kji == kji2)
    ss = rp.ssid.data
    resi = rp.p_resi.data
    resj = rp.p_resj.data
    assert np.all(np.right_shift(kij, 62) == ss[resi])
    assert np.all(np.right_shift(kji, 62) == ss[resj])
    assert np.all(np.right_shift(np.left_shift(kij, 2), 62) == ss[resj])
    assert np.all(np.right_shift(np.left_shift(kji, 2), 62) == ss[resi])


@pytest.mark.slow
def test_respairdat_addrots(respairdat):
    rotspace = get_rotamer_space()
    add_rots_to_respairdat(respairdat, rotspace)
    m, s = check_rotamer_deviation(respairdat, rotspace, quiet=1)
    assert m < 20
    assert s < 20
    add_xbin_to_respairdat(respairdat, min_ssep=10, cart_resl=1, ori_resl=20)


def test_remove_redundant_pdbs():
    pdbs = "12AS 155C 16PK 16VP 1914 19HC 1A04 1A05 1A0C 1A0D".split()
    assert 3 == len(remove_redundant_pdbs(pdbs, 30))
    assert 4 == len(remove_redundant_pdbs(pdbs, 40))
    assert 5 == len(remove_redundant_pdbs(pdbs, 50))
    assert 7 == len(remove_redundant_pdbs(pdbs, 70))
    assert 9 == len(remove_redundant_pdbs(pdbs, 90))
    assert 10 == len(remove_redundant_pdbs(pdbs, 95))


# if __name__ == "__main__":
#     # respairdat =
#     # test_jagged_bin()
#     test_jagged_bin_zero()
#     # test_respairdat()
