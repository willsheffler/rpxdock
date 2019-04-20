from time import perf_counter
import numpy as np
import homog as hm
from sicdock.body import Body


def test_body(C2_3hm4, C3_1nza, sym1=2, sym2=3):
    body1 = Body(C2_3hm4, sym1)
    body2 = Body(C3_1nza, sym2)

    resl = 5
    samp1 = range(0, 360 // sym1, resl)
    samp2 = range(0, 360 // sym2, resl)
    samp3 = range(-10, 11, resl)
    samp3 = [[np.cos(d / 180 * np.pi), 0, np.sin(d / 180 * np.pi)] for d in samp3]

    r1 = hm.hrot([0, 0, 1], 1, degrees=True)
    best, bestpos = -9e9, None
    t = perf_counter()
    totslide = 0
    nsamp, nhit = 0, 0
    for a1 in samp1:
        for a2 in samp2:
            for dirn in samp3:
                body1.move_to_center()
                body2.move_to_center()
                tmp = perf_counter()
                d = body1.slide_to(body2, dirn)
                totslide += perf_counter() - tmp
                nsamp += 1
                if d < 9e8:
                    nhit += 1
                    p = body1.cen_pairs(body2, 8.0)
                    if len(p) > 0:
                        p2 = body1.positioned_cen()[p[:, 0]]
                        p3 = body2.positioned_cen()[p[:, 1]]
                        assert np.max(np.linalg.norm(p3 - p2, axis=1)) < 8
                    if len(p) > best:
                        best = len(p)
                        bestpos = body1.pos.copy(), body2.pos.copy()
            body2.move_by(r1)
        body1.move_by(r1)
    t = perf_counter() - t
    print("best", best, "time", t, "rate", nsamp / t, "hitfrac", nhit / nsamp)
    print(
        "totslide",
        totslide,
        "slide/s",
        nsamp / totslide,
        "sqrt(npair)",
        np.sqrt(len(body1.ss) * len(body2.ss)),
        len(body1.ss),
        len(body2.ss),
    )
    # print(bestpos[0])
    # print(bestpos[1])
    body1.move_to(bestpos[0])
    body2.move_to(bestpos[1])
    # body1.dump_pdb_from_bodies("body1.pdb")
    # body2.dump_pdb_from_bodies("body2.pdb")


if __name__ == "__main__":
    import sicdock.rosetta as ros

    f1 = "sicdock/data/pdb/C2_3hm4_1.pdb.gz"
    f2 = "sicdock/data/pdb/C3_1nza_1.pdb.gz"
    # f1 = "/home/sheffler/scaffolds/big/C2_3jpz_1.pdb"
    # f2 = "/home/sheffler/scaffolds/big/C3_3ziy_1.pdb"
    # f1 = "/home/sheffler/scaffolds/wheel/C3.pdb"
    # f2 = "/home/sheffler/scaffolds/wheel/C5.pdb"
    pose1 = ros.get_pose_cached(f1)
    pose2 = ros.get_pose_cached(f2)
    test_body(pose1, pose2)

    # nres  306  309 sqnpair  307 new 17743/s orig 13511/s
    # nres  728 1371 sqnpair  999 new  8246/s orig  4287/s
    # nres 6675 8380 sqnpair 7479 new  8629/s orig   627/s
