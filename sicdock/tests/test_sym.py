from sicdock.sym import *
from homog import hdot


def test_sym():
    assert 33 in tetrahedral_axes
    assert 7 not in tetrahedral_axes
    for ax in symaxes.values():
        for a in ax.values():
            assert np.allclose(1, np.linalg.norm(a))

    print("neighboring component stuff")
    for sym in "TOI":
        assert np.allclose(symframes[sym][0], np.eye(4))
        for ax in symaxes[sym]:
            a = symaxes[sym][ax]
            dot = hdot(a, symframes[sym] @ a)
            mx = np.max(dot[dot < 0.9])
            w = np.where(np.abs(dot - mx) < 0.01)[0]
            print(sym, ax, w[0], mx)
            x = sym_to_neighbor_olig[sym][ax]
            assert np.allclose(hdot(a, x @ a), mx)


# for ide, bypass pytest
if __name__ == "__main__":
    test_sym()
