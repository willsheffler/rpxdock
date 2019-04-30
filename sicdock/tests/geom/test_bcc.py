import numpy as np
from sicdock.geom.bcc import *


def test_bcc_neighbors_3():

    for bcc in [
        BCC3([10, 10, 10], [-50, -50, -50], [50, 50, 50]),
        BCC3([11, 11, 11], [-55, -55, -55], [55, 55, 55]),
    ]:
        cen0 = np.array([[0.0, 0.0, 0.0]])
        kcen = bcc.keys(cen0)
        print(kcen)
        cen = bcc.vals(kcen)
        allkeys = np.arange(len(bcc), dtype="u8")
        allcens = bcc[allkeys]
        print(len(allcens))
        diff = allcens - cen
        d = np.linalg.norm(diff[:, :3], axis=1)

        for nexp in range(1, 5):
            nb = bcc.neighbors_3(kcen, nexp, 1, 0)
            wnb = set(nb)
            assert len(nb) == len(set(nb)) == (1 + 2 * nexp) ** 3 + 8 * nexp ** 3
            wd10 = set(np.where(d < 10.1 * nexp)[0])
            wd15 = set(np.where(d < 15.1 * nexp)[0])
            print(nexp, len(nb), len(wd15 - wnb), len(wnb - wd15))
            assert wd10.issubset(wnb)
            cart = bcc[nb]
            uvals = np.arange(-10 * nexp, 10.01 * nexp, 5)
            assert np.all(np.unique(cart[:, 0]) == uvals)
            assert np.all(np.unique(cart[:, 1]) == uvals)
            assert np.all(np.unique(cart[:, 2]) == uvals)

            # print(cart)
            com = np.mean(cart, axis=0)
            cerr = np.linalg.norm(com - cen)
            assert abs(cerr) < 0.001
            dis = np.linalg.norm(cart - com, axis=1)
            assert np.allclose(np.max(dis), np.sqrt(3) * nexp * 10)

        for nexp in range(1, 5):
            nb = bcc.neighbors_3(kcen, nexp, 1, 1)
            wnb = set(nb)
            assert len(nb) == len(set(nb)) == (1 + 2 * nexp) ** 3 + (2 * nexp + 2) ** 3
            wd10 = set(np.where(d < 10.1 * nexp)[0])
            wd15 = set(np.where(d < 15.1 * nexp)[0])
            print(nexp, len(nb), len(wd15 - wnb), len(wnb - wd15))
            assert wd10.issubset(wnb)
            cart = bcc[nb]
            uvals = np.arange(-10 * nexp - 5, 10.01 * nexp + 5, 5)
            assert np.all(np.unique(cart[:, 0]) == uvals)
            assert np.all(np.unique(cart[:, 1]) == uvals)
            assert np.all(np.unique(cart[:, 2]) == uvals)

            # print(cart)
            com = np.mean(cart, axis=0)
            cerr = np.linalg.norm(com - cen)
            assert abs(cerr) < 0.001
            dis = np.linalg.norm(cart - com, axis=1)
            assert np.allclose(np.max(dis), np.sqrt(3) * (nexp * 10 + 5))


def test_bcc_neighbors_6_3():

    bcc = BCC6(
        [10, 10, 10, 4, 4, 4], [-50, -50, -50, -20, -20, -20], [50, 50, 50, 20, 20, 20]
    )
    cen0 = np.array([[0.0, 0.0, 0.0, 0.5, 0.5, 0.5]])
    kcen = bcc.keys(cen0)
    cen = bcc.vals(kcen)

    allcens = bcc[np.arange(len(bcc), dtype="u8")]
    diff = allcens - cen
    d1 = np.linalg.norm(diff[:, :3], axis=1)
    d2 = np.linalg.norm(diff[:, 3:], axis=1)

    for nexp in range(1, 5):
        nb = bcc.neighbors_6_3(kcen, nexp, 1, 1)
        wnb = set(nb)
        assert len(nb) == len(set(nb)) == (1 + 2 * nexp) ** 3 + 64 * nexp ** 3

        wd = set(np.where((d1 < 10.1 * nexp + 5) * (d2 < 10))[0])
        assert len(wd - wnb) == 0
        vol_sph = 4 / 3 * np.pi
        vol_cube = 8
        cube_out_of_sphere = (vol_cube - vol_sph) / vol_cube
        assert len(wnb - wd) < len(wnb) * cube_out_of_sphere

        cart = bcc[nb]
        assert np.all(np.unique(cart[:, 0]) == np.arange(-10 * nexp, 10.01 * nexp, 5))
        assert np.all(np.unique(cart[:, 1]) == np.arange(-10 * nexp, 10.01 * nexp, 5))
        assert np.all(np.unique(cart[:, 2]) == np.arange(-10 * nexp, 10.01 * nexp, 5))
        assert np.all(np.unique(cart[:, 3]) == [-5, 0, 5])
        assert np.all(np.unique(cart[:, 4]) == [-5, 0, 5])
        assert np.all(np.unique(cart[:, 5]) == [-5, 0, 5])
        # print(cart)
        com = np.mean(cart, axis=0)
        cerr = np.linalg.norm(com - cen)
        assert abs(cerr) < 0.001
        dis = np.linalg.norm(cart - com, axis=1)
        assert np.allclose(np.max(dis), np.sqrt(3) * nexp * 10)


if __name__ == "__main__":
    # test_bcc_neighbors_3()
    test_bcc_neighbors_6_3()
