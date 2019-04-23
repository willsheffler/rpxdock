from sicdock.motif import *


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


def test_respairdat_mod(respairdat):
    rotspace = get_rotamer_space()
    add_rots_to_respairdat(respairdat, rotspace)
    m, s = check_rotamer_deviation(respairdat, rotspace, quiet=1)
    assert m < 20
    assert s < 20
    add_xbin_to_respairdat(respairdat, min_ssep=10, cart_resl=1, ori_resl=20)


# if __name__ == "__main__":
#     # respairdat =
#     # test_jagged_bin()
#     test_jagged_bin_zero()
#     # test_respairdat()
