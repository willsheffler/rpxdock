from sicdock.motif import *


def test_jagged_bin():
    kij = np.random.randint(1, 9e2, int(9e3)).astype("u8")
    kji = np.random.randint(1, 9e2, int(9e3)).astype("u8")
    k = np.concatenate([kij, kji])
    order, binkey, binrange = cpp.jagged_bin(k)
    kord = k[order]
    assert np.all(np.diff(kord) >= 0)
    lb = np.right_shift(binrange, 32)
    ub = binrange % 2 ** 32
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
    lb = np.right_shift(binrange, 32)
    ub = binrange % 2 ** 32
    assert np.sum(ub == lb) == 0, "not empty"
    assert np.all(kord[lb] == binkey)
    assert np.all(kord[ub - 1] == binkey)


if __name__ == "__main__":
    # test_jagged_bin()
    test_jagged_bin_zero()
