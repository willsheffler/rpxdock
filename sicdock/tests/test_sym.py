from sicdock.sym import *


def test_sym():
    assert 33 in tetrahedral_axes
    assert 7 not in tetrahedral_axes
    for ax in symaxes.values():
        for a in ax.values():
            assert np.allclose(1, np.linalg.norm(a))


# for ide, bypass pytest
if __name__ == "__main__":
    test_sym()
