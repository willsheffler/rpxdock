import numpy as np
import homog as hm
from sicdock.geom import BCC6
from sicdock.xbin import XBin
from sicdock.xbin import smear
from sicdock.phmap import PHMap_u8f8


def test_smear():
    xb = XBin(1.0, 9e9, 512)
    pm = PHMap_u8f8()
    # k = [xb.key_of([np.eye(4)])]
    k = xb.key_of(hm.rand_xform(1))
    cen = xb.bincen_of(k)
    # v = [0.0]
    # pm[k] = v
    # newmap = smear.smear(xb, pm)
    # print(pm[k])
    # print(newmap[k])
    nnb, unb = list(), list()
    maxdis = list()

    g = xb.grid()

    for i in range(1):
        nb = g.neighbors_6_3(k, 1, 0, 0)
        nnb.append(len(nb))
        unb.append(len(set(nb)))
        cart = g[nb][:, :]
        print(cart)
        # com = np.mean(cart, axis=0)
        # cerr = np.linalg.norm(com - cen[:, :3, 3])
        # print(com, cen[:, :3, 3])
        # assert cerr < 0.1
        # dis = np.linalg.norm(cart - com, axis=1)
        # maxdis.append(np.max(dis))
    # print(np.mean(nnb), np.mean(unb), np.max(maxdis))


if __name__ == "__main__":
    test_smear()
