import numpy as np
from rpxdock.geom.bcc import *

def test_bcc_neighbors_3():

   for bcc in [
         BCC3([10, 10, 10], [-50, -50, -50], [50, 50, 50]),
         BCC3([11, 11, 11], [-55, -55, -55], [55, 55, 55]),
   ]:
      cen0 = np.array([[0.0, 0.0, 0.0]])
      kcen = bcc.keys(cen0)
      # print(kcen)
      cen = bcc.vals(kcen)
      allkeys = np.arange(len(bcc), dtype="u8")
      allcens = bcc[allkeys]
      # print(len(allcens))
      diff = allcens - cen
      d = np.linalg.norm(diff[:, :3], axis=1)

      for rad in range(1, 5):
         nb = bcc.neighbors_3(kcen, rad, extrahalf=0, sphere=0).astype("i8")
         assert np.all(np.diff(nb) > 0)
         wnb = set(nb)
         assert len(nb) == len(set(nb)) == (1 + 2 * rad)**3 + 8 * rad**3
         wd10 = set(np.where(d < 10.1 * rad)[0])
         # wd15 = set(np.where(d < 15.1 * rad)[0])
         # print(rad, len(nb), len(wd15 - wnb), len(wnb - wd15))
         assert wd10.issubset(wnb)
         cart = bcc[nb.astype("u8")]
         uvals = np.arange(-10 * rad, 10.01 * rad, 5)
         # print(np.unique(cart[:, 0]))
         assert np.all(np.unique(cart[:, 0]) == uvals)
         assert np.all(np.unique(cart[:, 1]) == uvals)
         assert np.all(np.unique(cart[:, 2]) == uvals)

         # print(cart)
         com = np.mean(cart, axis=0)
         cerr = np.linalg.norm(com - cen)
         assert abs(cerr) < 0.001
         dis = np.linalg.norm(cart - com, axis=1)
         assert np.allclose(np.max(dis), np.sqrt(3) * rad * 10)

def test_bcc_neighbors_3_exhalf():
   for bcc in [
         BCC3([10, 10, 10], [-50, -50, -50], [50, 50, 50]),
         BCC3([11, 11, 11], [-55, -55, -55], [55, 55, 55]),
   ]:
      cen0 = np.array([[0.0, 0.0, 0.0]])
      kcen = bcc.keys(cen0)
      # print(kcen)
      cen = bcc.vals(kcen)
      allkeys = np.arange(len(bcc), dtype="u8")
      allcens = bcc[allkeys]
      # print(len(allcens))
      diff = allcens - cen
      d = np.linalg.norm(diff[:, :3], axis=1)

      for rad in range(1, 5):
         nb = bcc.neighbors_3(kcen, rad, extrahalf=1, sphere=0)
         cart = bcc[nb]
         # print(np.unique(cart[:, 0]))
         assert len(nb) == len(set(nb)) == (1 + 2 * rad)**3 + (2 * rad + 2)**3
         wnb = set(nb)
         wd10 = set(np.where(d < 10.1 * rad)[0])
         # wd15 = set(np.where(d < 15.1 * rad)[0])
         # print(rad, len(nb), len(wd15 - wnb), len(wnb - wd15))
         assert wd10.issubset(wnb)

         uvals = np.arange(-10 * rad - 5, 10.01 * rad + 5, 5)
         assert np.all(np.unique(cart[:, 0]) == uvals)
         assert np.all(np.unique(cart[:, 1]) == uvals)
         assert np.all(np.unique(cart[:, 2]) == uvals)

         # print(cart)
         com = np.mean(cart, axis=0)
         cerr = np.linalg.norm(com - cen)
         assert abs(cerr) < 0.001
         dis = np.linalg.norm(cart - com, axis=1)
         assert np.allclose(np.max(dis), np.sqrt(3) * (rad * 10 + 5))

def test_bcc_neighbors_3_sphere():

   for bcc in [
         BCC3([10, 10, 10], [-50, -50, -50], [50, 50, 50]),
         BCC3([11, 11, 11], [-55, -55, -55], [55, 55, 55]),
   ]:
      cen0 = np.array([[0.0, 0.0, 0.0]])
      kcen = bcc.keys(cen0)
      cen = bcc.vals(kcen)
      allkeys = np.arange(len(bcc), dtype="u8")
      allcens = bcc[allkeys]
      diff = allcens - cen
      d = np.linalg.norm(diff[:, :3], axis=1)
      ntrim = np.array([0, 8, 5 * 8 + 12, 23 * 8 + 3 * 12, 57 * 8 + 3 * 12])
      radius = [14.142135623730, 24.494897427831, 34.641016151377, 44.721359549995]

      for rad in range(1, 5):
         nbns = bcc.neighbors_3(kcen, rad, extrahalf=0, sphere=0).astype("i8")
         nb = bcc.neighbors_3(kcen, rad, extrahalf=0, sphere=1).astype("i8")
         cart = bcc[nb.astype("u8")]

         # from rpxdock.io.io import dump_pdb_from_points

         # cart2 = bcc[nbns.astype("u8")]
         # nbnse = bcc.neighbors_3(kcen, rad, extrahalf=1, sphere=0)
         # nbe = bcc.neighbors_3(kcen, rad, extrahalf=1, sphere=1)
         # carte = bcc[nbe]
         # cart2e = bcc[nbnse]
         # dump_pdb_from_points("bcc_%i.pdb" % rad, cart2)
         # dump_pdb_from_points("bcc_%i_sph.pdb" % rad, cart)
         # dump_pdb_from_points("bcc_%iex.pdb" % rad, cart2e)
         # dump_pdb_from_points("bcc_%iex_sph.pdb" % rad, carte)

         assert np.all(np.diff(nb) > 0)
         wnb = set(nb)
         # print("Npts", rad, (len(nbns) - len(nb) - ntrim[rad]) / 8)
         assert len(nb) == (1 + 2 * rad)**3 + 8 * rad**3 - ntrim[rad]
         wd10 = set(np.where(d < 10.1 * rad)[0])
         # wd15 = set(np.where(d < 15.1 * rad)[0])
         # print(rad, len(nb), len(wd15 - wnb), len(wnb - wd15))
         assert wd10.issubset(wnb)

         uvals = np.arange(-10 * rad, 10.01 * rad, 5)
         # print(np.unique(cart[:, 0]))
         assert np.all(np.unique(cart[:, 0]) == uvals)
         assert np.all(np.unique(cart[:, 1]) == uvals)
         assert np.all(np.unique(cart[:, 2]) == uvals)

         # print(cart)
         com = np.mean(cart, axis=0)
         cerr = np.linalg.norm(com - cen)
         assert abs(cerr) < 0.001
         dis = np.linalg.norm(cart - com, axis=1)
         # print(rad * 10, np.max(dis), np.sqrt(2) * rad * 10)
         assert rad * 10 < np.max(dis) < np.sqrt(2) * rad * 10 + 0.01
         assert np.allclose(radius[rad - 1], np.max(dis))

def test_bcc_neighbors_3_exhalf_sphere():

   for bcc in [
         BCC3([10, 10, 10], [-50, -50, -50], [50, 50, 50]),
         BCC3([11, 11, 11], [-55, -55, -55], [55, 55, 55]),
   ]:
      cen0 = np.array([[0.0, 0.0, 0.0]])
      kcen = bcc.keys(cen0)
      cen = bcc.vals(kcen)
      allkeys = np.arange(len(bcc), dtype="u8")
      allcens = bcc[allkeys]
      diff = allcens - cen
      d = np.linalg.norm(diff[:, :3], axis=1)

      ntrim = np.array([0, 4 * 8, 11 * 8, 36 * 8 + 3 * 12, 79 * 8 + 3 * 12])
      radius = [17.320508075688775, 30.0, 38.40572873934304, 50.0]
      for rad in range(1, 5):
         nbns = bcc.neighbors_3(kcen, rad, extrahalf=1, sphere=0).astype("i8")
         nb = bcc.neighbors_3(kcen, rad, extrahalf=1, sphere=1).astype("i8")
         # print(len(nbns), len(nb))
         cart = bcc[nb.astype("u8")]
         # cart2 = bcc[nbns.astype("u8")]
         assert np.all(np.diff(nb) > 0)
         wnb = set(nb)
         # print("Npts", rad, (len(nbns) - len(nb) - ntrim[rad]) / 8)

         assert len(nb) == (1 + 2 * rad)**3 + (2 * rad + 2)**3 - ntrim[rad]

         wd10 = set(np.where(d < 10.1 * rad)[0])
         # wd15 = set(np.where(d < 15.1 * rad)[0])
         # print(rad, len(nb), len(wd15 - wnb), len(wnb - wd15))
         assert wd10.issubset(wnb)

         uvals = np.arange(-10 * rad - 5, 10.01 * rad + 5, 5)
         # print(np.unique(cart[:, 0]))
         assert np.all(np.unique(cart[:, 0]) == uvals)
         assert np.all(np.unique(cart[:, 1]) == uvals)
         assert np.all(np.unique(cart[:, 2]) == uvals)

         # print(cart)
         com = np.mean(cart, axis=0)
         cerr = np.linalg.norm(com - cen)
         assert abs(cerr) < 0.001
         dis = np.linalg.norm(cart - com, axis=1)
         # print(rad * 10, np.max(dis), np.sqrt(2) * rad * 10)
         assert rad * 10 + 5 < np.max(dis) < np.sqrt(2) * (rad * 10 + 5.01)
         assert np.allclose(radius[rad - 1], np.max(dis))

def test_bcc_neighbors_6_3():

   cen0 = np.array([[0.0, 0.0, 0.0, 0.5, 0.5, 0.5]])
   for bcc in [
         BCC6(
            [10, 10, 10, 4, 4, 4],
            [-50, -50, -50, -20, -20, -20],
            [50, 50, 50, 20, 20, 20],
         ),
         BCC6(
            [11, 11, 11, 5, 5, 5],
            [-55, -55, -55, -25, -25, -25],
            [55, 55, 55, 25, 25, 25],
         ),
   ]:
      kcen = bcc.keys(cen0)
      cen = bcc.vals(kcen)
      assert np.all(cen == 0)
      allcens = bcc[np.arange(len(bcc), dtype="u8")]
      diff = allcens - cen
      d1 = np.linalg.norm(diff[:, :3], axis=1)
      d2 = np.linalg.norm(diff[:, 3:], axis=1)

      for rad in range(1, 5):
         nb = bcc.neighbors_6_3(kcen, rad, extrahalf=0, oddlast3=0, sphere=0)
         diff = np.diff(nb.astype("i8"))
         # print(diff)
         assert np.all(diff > 0)
         wnb = set(nb)
         assert len(nb) == len(set(nb)) == ((1 + 2 * rad)**3 + (2 * rad)**3)

         wd = set(np.where((d1 < 10.1 * rad + 5) * (d2 < 1))[0])
         assert len(wd - wnb) == 0

         cart = bcc[nb]
         uvals = np.arange(-10 * rad, 10.01 * rad, 5)
         assert np.all(np.unique(cart[:, 0]) == uvals)
         assert np.all(np.unique(cart[:, 1]) == uvals)
         assert np.all(np.unique(cart[:, 2]) == uvals)
         # assert np.all(np.unique(cart[:, 3]) == [-5, 0, 5])
         # assert np.all(np.unique(cart[:, 4]) == [-5, 0, 5])
         # assert np.all(np.unique(cart[:, 5]) == [-5, 0, 5])
         # print(cart)
         com = np.mean(cart, axis=0)
         cerr = np.linalg.norm(com[:3] - cen[0, :3])
         assert abs(cerr) < 0.001
         # dis = np.linalg.norm(cart - com, axis=1)
         # assert np.allclose(np.max(dis), np.sqrt(3) * rad * 10)

def test_bcc_neighbors_6_3_extrahalf():

   cen0 = np.array([[0.0, 0.0, 0.0, 0.5, 0.5, 0.5]])
   for bcc in [
         BCC6(
            [10, 10, 10, 4, 4, 4],
            [-50, -50, -50, -20, -20, -20],
            [50, 50, 50, 20, 20, 20],
         ),
         BCC6(
            [11, 11, 11, 5, 5, 5],
            [-55, -55, -55, -25, -25, -25],
            [55, 55, 55, 25, 25, 25],
         ),
   ]:
      kcen = bcc.keys(cen0)
      cen = bcc.vals(kcen)
      assert np.all(cen == 0)
      allcens = bcc[np.arange(len(bcc), dtype="u8")]
      diff = allcens - cen
      d1 = np.linalg.norm(diff[:, :3], axis=1)
      d2 = np.linalg.norm(diff[:, 3:], axis=1)

      for rad in range(1, 5):
         w = 2 * rad + 1
         nb = bcc.neighbors_6_3(kcen, rad, extrahalf=1, oddlast3=0, sphere=0)
         diff = np.diff(nb.astype("i8"))
         # print(diff)
         assert np.all(diff > 0)
         wnb = set(nb)
         assert len(nb) == len(set(nb)) == w**3 + (w + 1)**3

         wd = set(np.where((d1 < 10.1 * rad + 5) * (d2 < 1))[0])
         assert len(wd - wnb) == 0

         cart = bcc[nb]
         uvals = np.arange(-5 - 10 * rad, 10 * rad + 5.1, 5)
         assert np.all(np.unique(cart[:, 0]) == uvals)
         assert np.all(np.unique(cart[:, 1]) == uvals)
         assert np.all(np.unique(cart[:, 2]) == uvals)
         assert list(np.unique(cart[:, 3])) in [[-5, 0], [0, 5]]
         assert list(np.unique(cart[:, 4])) in [[-5, 0], [0, 5]]
         assert list(np.unique(cart[:, 5])) in [[-5, 0], [0, 5]]
         com = np.mean(cart, axis=0)
         cerr = np.linalg.norm(com[:3] - cen[0, :3])
         assert abs(cerr) < 0.001
         # dis = np.linalg.norm(cart - com, axis=1)
         # assert np.allclose(np.max(dis), np.sqrt(3) * rad * 10)

def test_bcc_neighbors_6_3_oddlast3():

   cen0 = np.array([[0.0, 0.0, 0.0, 0.5, 0.5, 0.5]])
   for bcc in [
         BCC6(
            [10, 10, 10, 4, 4, 4],
            [-50, -50, -50, -20, -20, -20],
            [50, 50, 50, 20, 20, 20],
         ),
         BCC6(
            [11, 11, 11, 5, 5, 5],
            [-55, -55, -55, -25, -25, -25],
            [55, 55, 55, 25, 25, 25],
         ),
   ]:
      kcen = bcc.keys(cen0)
      cen = bcc.vals(kcen)
      assert np.all(cen == 0)
      allcens = bcc[np.arange(len(bcc), dtype="u8")]
      diff = allcens - cen
      d1 = np.linalg.norm(diff[:, :3], axis=1)
      d2 = np.linalg.norm(diff[:, 3:], axis=1)

      for rad in range(1, 5):
         w = 2 * rad + 1
         nb = bcc.neighbors_6_3(kcen, rad, extrahalf=0, oddlast3=1, sphere=0)
         wnb = set(nb)
         diff = np.diff(nb.astype("i8"))
         # print(diff)
         assert np.all(diff > 0)

         assert len(nb) == w**3 + (w - 1)**3 * 8

         wd = set(np.where((d1 < 10.1 * rad + 5) * (d2 < 9))[0])
         # print(len(wd), len(wnb))
         assert len(wd - wnb) == 0
         vol_sph = 4 / 3 * np.pi
         vol_cube = 8
         cube_out_of_sphere = (vol_cube - vol_sph) / vol_cube
         # print(len(wnb - wd) / len(wnb))
         assert len(wnb - wd) < len(wnb) * cube_out_of_sphere

         cart = bcc[nb]
         uvals = np.arange(-10 * rad, 10.01 * rad, 5)
         assert np.all(np.unique(cart[:, 0]) == uvals)
         assert np.all(np.unique(cart[:, 1]) == uvals)
         assert np.all(np.unique(cart[:, 2]) == uvals)
         # print(np.unique(cart[:, 3]))
         assert np.all(np.unique(cart[:, 3]) == [-5, 0, 5])
         assert np.all(np.unique(cart[:, 4]) == [-5, 0, 5])
         assert np.all(np.unique(cart[:, 5]) == [-5, 0, 5])
         # print(cart)
         com = np.mean(cart, axis=0)
         cerr = np.linalg.norm(com - cen)
         assert abs(cerr) < 0.001
         dis = np.linalg.norm(cart - com, axis=1)
         assert np.allclose(np.max(dis), np.sqrt(3) * rad * 10)

def test_bcc_neighbors_6_3_oddlast3_extrahalf():
   radius = [
      0,
      27.386127875258307,
      44.15880433163923,
      61.237243569579455,
      78.4219357067906,
   ]
   cen0 = np.array([[0.0, 0.0, 0.0, 0.5, 0.5, 0.5]])
   for bcc in [
         BCC6(
            [10, 10, 10, 4, 4, 4],
            [-50, -50, -50, -20, -20, -20],
            [50, 50, 50, 20, 20, 20],
         ),
         BCC6(
            [11, 11, 11, 5, 5, 5],
            [-55, -55, -55, -25, -25, -25],
            [55, 55, 55, 25, 25, 25],
         ),
   ]:
      kcen = bcc.keys(cen0)
      cen = bcc.vals(kcen)
      assert np.all(cen == 0)
      allcens = bcc[np.arange(len(bcc), dtype="u8")]
      diff = allcens - cen
      d1 = np.linalg.norm(diff[:, :3], axis=1)
      d2 = np.linalg.norm(diff[:, 3:], axis=1)

      for rad in range(1, 5):
         w = 2 * rad + 1
         nb = bcc.neighbors_6_3(kcen, rad, extrahalf=1, oddlast3=1, sphere=0)
         wnb = set(nb)
         diff = np.diff(nb.astype("i8"))
         # print(diff)
         assert np.all(diff > 0)

         assert len(nb) == w**3 + (w + 1)**3 * 8

         wd = set(np.where((d1 < 10.1 * rad + 9) * (d2 < 9))[0])
         # print(len(wd), len(wnb))
         assert len(wd - wnb) == 0
         assert len(wnb - wd) < len(wnb) * 0.6

         cart = bcc[nb]
         uvals = np.arange(-5 - 10 * rad, 5.01 + 10 * rad, 5)
         assert np.all(np.unique(cart[:, 0]) == uvals)
         assert np.all(np.unique(cart[:, 1]) == uvals)
         assert np.all(np.unique(cart[:, 2]) == uvals)
         # print(np.unique(cart[:, 3]))
         assert np.all(np.unique(cart[:, 3]) == [-5, 0, 5])
         assert np.all(np.unique(cart[:, 4]) == [-5, 0, 5])
         assert np.all(np.unique(cart[:, 5]) == [-5, 0, 5])
         # print(cart)
         com = np.mean(cart, axis=0)
         cerr = np.linalg.norm(com - cen)
         assert abs(cerr) < 0.001
         dis = np.linalg.norm(cart - com, axis=1)
         # assert np.allclose(np.max(dis), np.sqrt(3) * rad * 10)
         # print(rad * 10 + 10, np.max(dis), np.sqrt(3) * (rad * 10 + 10.01))
         assert rad * 10 + 10 < np.max(dis) < np.sqrt(3) * (rad * 10 + 10.01)
         assert np.allclose(radius[rad], np.max(dis))

def test_bcc_neighbors_6_3_oddlast3_sphere():
   ntrim = np.array([0, 8, 12 * 8 + 12, 72 * 8 + 3 * 12, 190 * 8 + 3 * 12])
   radius = [
      0,
      14.142135623730951,
      24.49489742783178,
      34.64101615137755,
      44.721359549995796,
   ]

   cen0 = np.array([[0.0, 0.0, 0.0, 0.5, 0.5, 0.5]])
   for bcc in [
         BCC6(
            [10, 10, 10, 4, 4, 4],
            [-50, -50, -50, -20, -20, -20],
            [50, 50, 50, 20, 20, 20],
         ),
         BCC6(
            [11, 11, 11, 5, 5, 5],
            [-55, -55, -55, -25, -25, -25],
            [55, 55, 55, 25, 25, 25],
         ),
   ]:
      kcen = bcc.keys(cen0)
      cen = bcc.vals(kcen)
      assert np.all(cen == 0)
      allcens = bcc[np.arange(len(bcc), dtype="u8")]
      diff = allcens - cen
      d1 = np.linalg.norm(diff[:, :3], axis=1)
      d2 = np.linalg.norm(diff[:, 3:], axis=1)

      for rad in range(1, 5):
         w = 2 * rad + 1
         nb = bcc.neighbors_6_3(kcen, rad, extrahalf=0, oddlast3=1, sphere=1)
         nbns = bcc.neighbors_6_3(kcen, rad, extrahalf=0, oddlast3=1, sphere=0)
         assert len(nb) < len(nbns)
         wnb = set(nb)
         diff = np.diff(nb.astype("i8"))
         # print(diff)
         assert np.all(diff > 0)

         print(
            "test_bcc_neighbors_6_3_oddlast3_sphere",
            rad,
            len(nb),
            len(nbns) - len(nb),
         )
         assert len(nbns) == w**3 + (w - 1)**3 * 8
         assert len(nb) == w**3 + (w - 1)**3 * 8 - ntrim[rad]

         wd = set(np.where((d1 < 10.1 * rad + 5) * (d2 < 9))[0])
         # print(len(wd), len(wnb))
         assert len(wd - wnb) == 0
         vol_sph = 4 / 3 * np.pi
         vol_cube = 8
         cube_out_of_sphere = (vol_cube - vol_sph) / vol_cube
         # print(len(wnb - wd) / len(wnb))
         assert len(wnb - wd) < len(wnb) * cube_out_of_sphere

         cart = bcc[nb]
         uvals = np.arange(-10 * rad, 10.01 * rad, 5)
         assert np.all(np.unique(cart[:, 0]) == uvals)
         assert np.all(np.unique(cart[:, 1]) == uvals)
         assert np.all(np.unique(cart[:, 2]) == uvals)
         # print(np.unique(cart[:, 3]))
         assert np.all(np.unique(cart[:, 3]) == [-5, 0, 5])
         assert np.all(np.unique(cart[:, 4]) == [-5, 0, 5])
         assert np.all(np.unique(cart[:, 5]) == [-5, 0, 5])
         # print(cart)
         com = np.mean(cart, axis=0)
         cerr = np.linalg.norm(com - cen)
         assert abs(cerr) < 0.001
         dis = np.linalg.norm(cart - com, axis=1)
         assert rad * 10 < np.max(dis) < np.sqrt(2) * rad * 10 + 0.01
         assert np.allclose(radius[rad], np.max(dis))

def test_bcc_neighbors_6_3_oddlast3_sphere_extrahalf():
   ntrim = np.array([0, 4 * 8 * 8, 81 * 8, 239 * 8 + 3 * 12, 471 * 8 + 3 * 12])
   radius = [
      0,
      18.708286933869708,
      30.822070014844883,
      39.370039370059054,
      50.49752469181039,
   ]
   cen0 = np.array([[0.0, 0.0, 0.0, 0.5, 0.5, 0.5]])
   for bcc in [
         BCC6(
            [10, 10, 10, 4, 4, 4],
            [-50, -50, -50, -20, -20, -20],
            [50, 50, 50, 20, 20, 20],
         ),
         BCC6(
            [11, 11, 11, 5, 5, 5],
            [-55, -55, -55, -25, -25, -25],
            [55, 55, 55, 25, 25, 25],
         ),
   ]:
      kcen = bcc.keys(cen0)
      cen = bcc.vals(kcen)
      assert np.all(cen == 0)
      allcens = bcc[np.arange(len(bcc), dtype="u8")]
      diff = allcens - cen
      d1 = np.linalg.norm(diff[:, :3], axis=1)
      d2 = np.linalg.norm(diff[:, 3:], axis=1)

      for rad in range(1, 5):
         w = 2 * rad + 1
         nb = bcc.neighbors_6_3(kcen, rad, extrahalf=1, oddlast3=1, sphere=1)
         nbns = bcc.neighbors_6_3(kcen, rad, extrahalf=1, oddlast3=1, sphere=0)
         cart = bcc[nb]
         cart2 = bcc[nbns]
         assert len(nb) < len(nbns)
         wnb = set(nb)
         diff = np.diff(nb.astype("i8"))
         assert np.all(diff > 0)

         assert len(nbns) == w**3 + (w + 1)**3 * 8
         assert len(nb) == w**3 + (w + 1)**3 * 8 - ntrim[rad]

         wd = set(np.where((d1 < 10.1 * rad + 9) * (d2 < 9))[0])
         # print(len(wd), len(wnb))
         assert len(wd - wnb) == 0
         vol_sph = 4 / 3 * np.pi
         vol_cube = 8
         cube_out_of_sphere = (vol_cube - vol_sph) / vol_cube
         # print(len(wnb - wd) / len(wnb))
         assert len(wnb - wd) < len(wnb) * cube_out_of_sphere

         uvals = np.arange(-5 - 10 * rad, 5 + 10.01 * rad, 5)
         assert np.all(np.unique(cart[:, 0]) == uvals)
         assert np.all(np.unique(cart[:, 1]) == uvals)
         assert np.all(np.unique(cart[:, 2]) == uvals)
         # print(np.unique(cart[:, 3]))
         assert np.all(np.unique(cart[:, 3]) == [-5, 0, 5])
         assert np.all(np.unique(cart[:, 4]) == [-5, 0, 5])
         assert np.all(np.unique(cart[:, 5]) == [-5, 0, 5])
         # print(cart)
         com = np.mean(cart, axis=0)
         # print(com)
         cerr = np.linalg.norm(com - cen)
         assert abs(cerr) < 0.001
         dis = np.linalg.norm(cart - com, axis=1)
         assert 5 + rad * 10 < np.max(dis) < np.sqrt(2) * (rad * 10 + 5.01)
         print(np.max(dis))
         # assert np.allclose(radius[rad], np.max(dis))

def test_bcc_neighbor_radous():
   # no bounts checking
   bcc = BCC6([10, 10, 10, 10, 10, 10], [-1, -1, -1, -1, -1, -1], [1, 1, 1, 1, 1, 1])
   kcen = bcc.keys(np.zeros(6).reshape(1, 6))
   cen = bcc.vals(kcen)
   for r, e, s, c in [
      (1, 0, 10, 12 + 1),
      (1, 1, 17, 27 + 1),
      (2, 0, 26, 48 + 1),
      (2, 1, 37, 75 + 1),
      (3, 0, 50, 108 + 1),
      (3, 1, 65, 147 + 1),
      (4, 0, 82, 192 + 1),
      (4, 1, 101, 243 + 1),
      (5, 0, 122, 300 + 1),
      (5, 1, 145, 363 + 1),
      (6, 0, 170, 432 + 1),
      (6, 1, 197, 507 + 1),
   ]:
      s0 = bcc.neighbor_sphere_radius_square_cut(radius=r, extrahalf=e)
      c0 = bcc.neighbor_radius_square_cut(radius=r, extrahalf=e)
      assert s0 == s
      assert c0 == c
      ks, ds = bcc.neighbors_6_3_dist(kcen, r, extrahalf=e, oddlast3=1, sphere=1)
      kc, dc = bcc.neighbors_6_3_dist(kcen, r, extrahalf=e, oddlast3=1, sphere=0)
      print(s, np.max(ds))
      if r > 1:
         assert s * 0.9 < np.max(ds)
      assert np.max(ds) < s
      assert c * 0.9 < np.max(dc) < c

if __name__ == "__main__":
   # test_bcc_neighbors_3()
   # test_bcc_neighbors_3_sphere()
   # test_bcc_neighbors_3_exhalf_sphere()
   # test_bcc_neighbors_6_3()
   # test_bcc_neighbors_6_3_extrahalf()
   # test_bcc_neighbors_6_3_oddlast3()
   # test_bcc_neighbors_6_3_oddlast3_extrahalf()
   # test_bcc_neighbors_6_3_oddlast3_sphere()
   # test_bcc_neighbors_6_3_oddlast3_sphere_extrahalf()
   test_bcc_neighbor_radous()
