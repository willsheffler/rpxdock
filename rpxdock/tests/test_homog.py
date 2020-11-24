import rpxdock as rp, pytest, numpy as np, itertools as it, functools as ft

from rpxdock.homog import *
import rpxdock.homog as hm

def test_sym():
   assert rp.geom.sym.tetrahedral_frames.shape == (12, 4, 4)
   assert rp.geom.sym.octahedral_frames.shape == (24, 4, 4)
   assert rp.geom.sym.icosahedral_frames.shape == (60, 4, 4)
   x = np.concatenate([
      rp.geom.sym.tetrahedral_frames,
      rp.geom.sym.octahedral_frames,
      rp.geom.sym.icosahedral_frames,
   ])
   assert np.all(x[..., 3, 3] == 1)
   assert np.all(x[..., 3, :3] == 0)
   assert np.all(x[..., :3, 3] == 0)

def test_homo_rotation_single():
   axis0 = hnormalized(np.random.randn(3))
   ang0 = np.pi / 4.0
   r = hrot(list(axis0), float(ang0))
   a = fast_axis_of(r)
   n = hnorm(a)
   assert np.all(abs(a / n - axis0) < 0.001)
   assert np.all(abs(np.arcsin(n / 2) - ang0) < 0.001)

def test_homo_rotation_center():
   assert np.allclose([0, 2, 0, 1], hrot([1, 0, 0], 180, [0, 1, 0]) @ (0, 0, 0, 1), atol=1e-5)
   assert np.allclose([0, 1, -1, 1], hrot([1, 0, 0], 90, [0, 1, 0]) @ (0, 0, 0, 1), atol=1e-5)
   assert np.allclose([-1, 1, 2, 1], hrot([1, 1, 0], 180, [0, 1, 1]) @ (0, 0, 0, 1), atol=1e-5)

def test_homo_rotation_array():
   shape = (1, 2, 1, 3, 4, 1, 1)
   axis0 = hnormalized(np.random.randn(*(shape + (3, ))))
   ang0 = np.random.rand(*shape) * (0.99 * np.pi / 2 + 0.005 * np.pi / 2)
   r = hrot(axis0, ang0)
   a = fast_axis_of(r)
   n = hnorm(a)[..., np.newaxis]
   assert np.all(abs(a / n - axis0) < 0.001)
   assert np.all(abs(np.arcsin(n[..., 0] / 2) - ang0) < 0.001)

def test_homo_rotation_angle():
   ang = np.random.rand(1000) * np.pi
   a = rand_unit()
   u = proj_perp(a, rand_vec())
   x = hrot(a, ang)
   ang2 = angle(u, x @ u)
   assert np.allclose(ang, ang2, atol=1e-5)

def test_htrans():
   assert htrans([1, 3, 7]).shape == (4, 4)
   assert np.allclose(htrans([1, 3, 7])[:3, 3], (1, 3, 7))

   with pytest.raises(ValueError):
      htrans([4, 3, 2, 1])

   s = (2, )
   t = np.random.randn(*s, 3)
   ht = htrans(t)
   assert ht.shape == s + (4, 4)
   assert np.allclose(ht[..., :3, 3], t)

def test_hcross():
   assert np.allclose(hcross([1, 0, 0], [0, 1, 0]), [0, 0, 1])
   assert np.allclose(hcross([1, 0, 0, 0], [0, 1, 0, 0]), [0, 0, 1, 0])
   a, b = np.random.randn(3, 4, 5, 3), np.random.randn(3, 4, 5, 3)
   c = hcross(a, b)
   assert np.allclose(hdot(a, c), 0)
   assert np.allclose(hdot(b, c), 0)

def test_axis_angle_of():
   ax, an = axis_angle_of(hrot([10, 10, 0], np.pi))
   assert 1e-5 > np.abs(ax[0] - ax[1])
   assert 1e-5 > np.abs(ax[2])
   ax, an = axis_angle_of(hrot([0, 1, 0], np.pi))
   assert 1e-5 > np.abs(ax[0])
   assert 1e-5 > np.abs(ax[1]) - 1
   assert 1e-5 > np.abs(ax[2])

   ax, an = axis_angle_of(hrot([0, 1, 0], np.pi * 0.25))
   print(ax, an)
   assert np.allclose(ax, [0, 1, 0, 0], atol=1e-5)
   assert 1e-5 > np.abs(an - np.pi * 0.25)
   ax, an = axis_angle_of(hrot([0, 1, 0], np.pi * 0.75))
   print(ax, an)
   assert np.allclose(ax, [0, 1, 0, 0], atol=1e-5)
   assert 1e-5 > np.abs(an - np.pi * 0.75)

   ax, an = axis_angle_of(hrot([1, 0, 0], np.pi / 2))
   print(np.pi / an)
   assert 1e-5 > np.abs(an - np.pi / 2)

def test_axis_angle_of_rand():
   shape = (4, 5, 6, 7, 8)
   axis = hnormalized(np.random.randn(*shape, 3))
   angl = np.random.random(shape) * np.pi / 2
   rot = hrot(axis, angl, dtype='f8')
   ax, an = axis_angle_of(rot)
   assert np.allclose(axis, ax, atol=1e-3, rtol=1e-3)  # very loose to allow very rare cases
   assert np.allclose(angl, an, atol=1e-4, rtol=1e-4)

def test_is_valid_rays():
   assert not is_valid_rays([[0, 1], [0, 0], [0, 0], [0, 0]])
   assert not is_valid_rays([[0, 0], [0, 0], [0, 0], [1, 0]])
   assert not is_valid_rays([[0, 0], [0, 3], [0, 0], [1, 0]])
   assert is_valid_rays([[0, 0], [0, 1], [0, 0], [1, 0]])

def test_rand_ray():
   r = rand_ray()
   assert np.all(r[..., 3, :] == (1, 0))
   assert r.shape == (4, 2)
   assert np.allclose(hnorm(r[..., :3, 1]), 1)

   r = rand_ray(shape=(5, 6, 7))
   assert np.all(r[..., 3, :] == (1, 0))
   assert r.shape == (5, 6, 7, 4, 2)
   assert np.allclose(hnorm(r[..., :3, 1]), 1)

def test_proj_prep():
   assert np.allclose([2, 3, 0], proj_perp([0, 0, 1], [2, 3, 99]))
   assert np.allclose([2, 3, 0], proj_perp([0, 0, 2], [2, 3, 99]))
   a, b = np.random.randn(2, 5, 6, 7, 3)
   pp = proj_perp(a, b)
   assert np.allclose(hdot(a, pp), 0, atol=1e-5)

def test_point_in_plane():
   plane = rand_ray((5, 6, 7))
   assert np.all(point_in_plane(plane, plane[..., :3, 0]))
   pt = proj_perp(plane[..., :3, 1], np.random.randn(3))
   assert np.all(point_in_plane(plane, plane[..., :3, 0] + pt))

def test_ray_in_plane():
   plane = rand_ray((5, 6, 7))
   dirn = proj_perp(plane[..., :3, 1], np.random.randn(5, 6, 7, 3))
   ray = hray(plane[..., :3, 0] + np.cross(plane[..., :3, 1], dirn) * 7, dirn)
   assert np.all(ray_in_plane(plane, ray))

def test_intersect_planes():
   with pytest.raises(ValueError):
      intersect_planes(
         np.array([[0, 0, 0, 2], [0, 0, 0, 0]]).T,
         np.array([[0, 0, 0, 1], [0, 0, 0, 0]]).T)
   with pytest.raises(ValueError):
      intersect_planes(
         np.array([[0, 0, 0, 1], [0, 0, 0, 0]]).T,
         np.array([[0, 0, 0, 1], [0, 0, 0, 1]]).T)
   with pytest.raises(ValueError):
      intersect_planes(
         np.array([[0, 0, 1], [0, 0, 0, 0]]).T,
         np.array([[0, 0, 1], [0, 0, 0, 1]]).T)
   with pytest.raises(ValueError):
      intersect_planes(np.array(9 * [[[0, 0], [0, 0], [0, 0], [1, 0]]]),
                       np.array(2 * [[[0, 0], [0, 0], [0, 0], [1, 0]]]))

   # isct, sts = intersect_planes(np.array(9 * [[[0, 0, 0, 1], [1, 0, 0, 0]]]),
   # np.array(9 * [[[0, 0, 0, 1], [1, 0, 0, 0]]]))
   # assert isct.shape[:-2] == sts.shape == (9,)
   # assert np.all(sts == 2)

   # isct, sts = intersect_planes(np.array([[1, 0, 0, 1], [1, 0, 0, 0]]),
   # np.array([[0, 0, 0, 1], [1, 0, 0, 0]]))
   # assert sts == 1

   isct, sts = intersect_planes(
      np.array([[0, 0, 0, 1], [1, 0, 0, 0]]).T,
      np.array([[0, 0, 0, 1], [0, 1, 0, 0]]).T)
   assert sts == 0
   assert isct[2, 0] == 0
   assert np.all(abs(isct[:3, 1]) == (0, 0, 1))

   isct, sts = intersect_planes(
      np.array([[0, 0, 0, 1], [1, 0, 0, 0]]).T,
      np.array([[0, 0, 0, 1], [0, 0, 1, 0]]).T)
   assert sts == 0
   assert isct[1, 0] == 0
   assert np.all(abs(isct[:3, 1]) == (0, 1, 0))

   isct, sts = intersect_planes(
      np.array([[0, 0, 0, 1], [0, 1, 0, 0]]).T,
      np.array([[0, 0, 0, 1], [0, 0, 1, 0]]).T)
   assert sts == 0
   assert isct[0, 0] == 0
   assert np.all(abs(isct[:3, 1]) == (1, 0, 0))

   isct, sts = intersect_planes(
      np.array([[7, 0, 0, 1], [1, 0, 0, 0]]).T,
      np.array([[0, 9, 0, 1], [0, 1, 0, 0]]).T)
   assert sts == 0
   assert np.allclose(isct[:3, 0], [7, 9, 0])
   assert np.allclose(abs(isct[:3, 1]), [0, 0, 1])

   isct, sts = intersect_planes(
      np.array([[0, 0, 0, 1], hnormalized([1, 1, 0, 0])]).T,
      np.array([[0, 0, 0, 1], hnormalized([0, 1, 1, 0])]).T)
   assert sts == 0
   assert np.allclose(abs(isct[:, 1]), hnormalized([1, 1, 1]))

   p1 = hray([2, 0, 0, 1], [1, 0, 0, 0])
   p2 = hray([0, 0, 0, 1], [0, 0, 1, 0])
   isct, sts = intersect_planes(p1, p2)
   assert sts == 0
   assert np.all(ray_in_plane(p1, isct))
   assert np.all(ray_in_plane(p2, isct))

   p1 = np.array([[0.39263901, 0.57934885, -0.7693232, 1.],
                  [-0.80966465, -0.18557869, 0.55677976, 0.]]).T
   p2 = np.array([[0.14790894, -1.333329, 0.45396509, 1.],
                  [-0.92436319, -0.0221499, 0.38087016, 0.]]).T
   isct, sts = intersect_planes(p1, p2)
   assert sts == 0
   assert np.all(ray_in_plane(p1, isct))
   assert np.all(ray_in_plane(p2, isct))

def test_intersect_planes_rand():
   # origin case
   plane1, plane2 = rand_ray(shape=(2, 1))
   plane1[..., :3, 0] = 0
   plane2[..., :3, 0] = 0
   isect, status = intersect_planes(plane1, plane2)
   assert np.all(status == 0)
   assert np.all(ray_in_plane(plane1, isect))
   assert np.all(ray_in_plane(plane2, isect))

   # orthogonal case
   plane1, plane2 = rand_ray(shape=(2, 1))
   plane1[..., :, 1] = hnormalized([0, 0, 1])
   plane2[..., :, 1] = hnormalized([0, 1, 0])
   isect, status = intersect_planes(plane1, plane2)
   assert np.all(status == 0)
   assert np.all(ray_in_plane(plane1, isect))
   assert np.all(ray_in_plane(plane2, isect))

   # general case
   plane1, plane2 = rand_ray(shape=(2, 5, 6, 7, 8, 9))
   isect, status = intersect_planes(plane1, plane2)
   assert np.all(status == 0)
   assert np.all(ray_in_plane(plane1, isect))
   assert np.all(ray_in_plane(plane2, isect))

def test_axis_ang_cen_of_rand():
   shape = (5, 6, 7, 8, 9)
   axis0 = hnormalized(np.random.randn(*shape, 3))
   ang0 = np.random.random(shape) * (np.pi - 0.1) + 0.1
   cen0 = np.random.randn(*shape, 3) * 100.0

   helical_trans = np.random.randn(*shape)[..., None] * axis0
   rot = hrot(axis0, ang0, cen0, dtype='f8')
   rot[..., :, 3] += helical_trans
   axis, ang, cen = axis_ang_cen_of(rot)

   assert np.allclose(axis0, axis, rtol=1e-5)
   assert np.allclose(ang0, ang, rtol=1e-5)
   #  check rotation doesn't move cen
   cenhat = (rot @ cen[..., None]).squeeze()
   assert np.allclose(cen + helical_trans, cenhat, rtol=1e-4, atol=1e-4)

def test_axis_angle_vs_axis_angle_cen_performance():
   N = 10000
   t = rp.Timer().start()
   xforms = rand_xform(N)
   t.checkpoint('setup')
   axis, ang = axis_angle_of(xforms)
   t.checkpoint('axis_angle_of')
   axis2, ang2, cen = axis_ang_cen_of(xforms)
   t.checkpoint('axis_ang_cen_of')
   print(t.report(scale=1_000_000 / N))

   assert np.allclose(axis, axis2)
   assert np.allclose(ang, ang2)
   # assert 0

def test_hinv_rand():
   shape = (5, 6, 7, 8, 9)
   axis0 = hnormalized(np.random.randn(*shape, 3))
   ang0 = np.random.random(shape) * (np.pi - 0.1) + 0.1
   cen0 = np.random.randn(*shape, 3) * 100.0
   helical_trans = np.random.randn(*shape)[..., None] * axis0
   rot = hrot(axis0, ang0, cen0, dtype='f8')
   rot[..., :, 3] += helical_trans
   assert np.allclose(np.eye(4), hinv(rot) @ rot)

def test_hstub():
   sh = (5, 6, 7, 8, 9)
   u = h_rand_points(sh)
   v = h_rand_points(sh)
   w = h_rand_points(sh)
   s = hstub(u, v, w)
   assert is_homog_xform(s)

   assert is_homog_xform(hstub([1, 2, 3], [5, 6, 4], [9, 7, 8]))

def test_line_line_dist():
   lld = line_line_distance
   assert lld(hray([0, 0, 0], [1, 0, 0]), hray([0, 0, 0], [1, 0, 0])) == 0
   assert lld(hray([0, 0, 0], [1, 0, 0]), hray([1, 0, 0], [1, 0, 0])) == 0
   assert lld(hray([0, 0, 0], [1, 0, 0]), hray([0, 1, 0], [1, 0, 0])) == 1
   assert lld(hray([0, 0, 0], [1, 0, 0]), hray([0, 1, 0], [0, 0, 1])) == 1

def test_line_line_closest_points():
   lld = line_line_distance
   llcp = line_line_closest_points
   p, q = llcp(hray([0, 0, 0], [1, 0, 0]), hray([0, 0, 0], [0, 1, 0]))
   assert np.all(p == [0, 0, 0, 1]) and np.all(q == [0, 0, 0, 1])
   p, q = llcp(hray([0, 1, 0], [1, 0, 0]), hray([1, 0, 0], [0, 1, 0]))
   assert np.all(p == [1, 1, 0, 1]) and np.all(q == [1, 1, 0, 1])
   p, q = llcp(hray([1, 1, 0], [1, 0, 0]), hray([1, 1, 0], [0, 1, 0]))
   assert np.all(p == [1, 1, 0, 1]) and np.all(q == [1, 1, 0, 1])
   p, q = llcp(hray([1, 2, 3], [1, 0, 0]), hray([4, 5, 6], [0, 1, 0]))
   assert np.all(p == [4, 2, 3, 1]) and np.all(q == [4, 2, 6, 1])
   p, q = llcp(hray([1, 2, 3], [-13, 0, 0]), hray([4, 5, 6], [0, -7, 0]))
   assert np.all(p == [4, 2, 3, 1]) and np.all(q == [4, 2, 6, 1])
   p, q = llcp(hray([1, 2, 3], [1, 0, 0]), hray([4, 5, 6], [0, 1, 0]))
   assert np.all(p == [4, 2, 3, 1]) and np.all(q == [4, 2, 6, 1])

   r1, r2 = hray([1, 2, 3], [1, 0, 0]), hray([4, 5, 6], [0, 1, 0])
   x = rand_xform((5, 6, 7))
   xinv = np.linalg.inv(x)
   p, q = llcp(x @ r1, x @ r2)
   assert np.allclose((xinv @ p[..., None]).squeeze(-1), [4, 2, 3, 1])
   assert np.allclose((xinv @ q[..., None]).squeeze(-1), [4, 2, 6, 1])

   shape = (5, 6, 7)
   r1 = rand_ray(cen=np.random.randn(*shape, 3))
   r2 = rand_ray(cen=np.random.randn(*shape, 3))
   p, q = llcp(r1, r2)
   assert p.shape[:-1] == shape and q.shape[:-1] == shape
   lldist0 = hnorm(p - q)
   lldist1 = lld(r1, r2)
   # print(lldist0 - lldist1)
   # TODO figure out how to compare better
   assert np.allclose(lldist0, lldist1, atol=1e-1, rtol=1e-1)  # loose, but rarely fails otherwise

def test_dihedral():
   assert 0.00001 > abs(np.pi / 2 - dihedral([1, 0, 0], [0, 1, 0], [0, 0, 0], [0, 0, 1]))
   assert 0.00001 > abs(-np.pi / 2 - dihedral([1, 0, 0], [0, 0, 0], [0, 1, 0], [0, 0, 1]))
   a, b, c = hpoint([1, 0, 0]), hpoint([0, 1, 0]), hpoint([0, 0, 1]),
   n = hpoint([0, 0, 0])
   x = rand_xform(10)
   assert np.allclose(dihedral(a, b, c, n), dihedral(x @ a, x @ b, x @ c, x @ n))
   for ang in np.arange(-np.pi + 0.001, np.pi, 0.1):
      x = hrot([0, 1, 0], ang)
      d = dihedral([1, 0, 0], [0, 0, 0], [0, 1, 0], x @ [1, 0, 0, 0])
      assert abs(ang - d) < 0.000001

def test_angle():
   assert 0.0001 > abs(angle([1, 0, 0], [0, 1, 0]) - np.pi / 2)
   assert 0.0001 > abs(angle([1, 1, 0], [0, 1, 0]) - np.pi / 4)

def test_align_around_axis():
   axis = rand_unit(1000)
   u = rand_vec()
   ang = np.random.rand(1000) * np.pi
   x = hrot(axis, ang)
   v = x @ u
   uprime = align_around_axis(axis, u, v) @ u
   assert np.allclose(angle(v, uprime), 0, atol=1e-5)

def test_align_vectors_minangle():

   tgt1 = [-0.816497, -0.000000, -0.577350, 0]
   tgt2 = [0.000000, 0.000000, 1.000000, 0]
   orig1 = [0.000000, 0.000000, 1.000000, 0]
   orig2 = [-0.723746, 0.377967, -0.577350, 0]
   x = align_vectors(orig1, orig2, tgt1, tgt2)
   assert np.allclose(tgt1, x @ orig1, atol=1e-5)
   assert np.allclose(tgt2, x @ orig2, atol=1e-5)

   ax1 = np.array([0.12896027, -0.57202471, -0.81003518, 0.])
   ax2 = np.array([0., 0., -1., 0.])
   tax1 = np.array([0.57735027, 0.57735027, 0.57735027, 0.])
   tax2 = np.array([0.70710678, 0.70710678, 0., 0.])
   x = align_vectors(ax1, ax2, tax1, tax2)
   assert np.allclose(x @ ax1, tax1, atol=1e-2)
   assert np.allclose(x @ ax2, tax2, atol=1e-2)

def test_align_vectors_una_case():
   ax1 = np.array([0., 0., -1., 0.])
   ax2 = np.array([0.83822463, -0.43167392, 0.33322229, 0.])
   tax1 = np.array([-0.57735027, 0.57735027, 0.57735027, 0.])
   tax2 = np.array([0.57735027, -0.57735027, 0.57735027, 0.])
   # print(angle_degrees(ax1, ax2))
   # print(angle_degrees(tax1, tax2))
   x = align_vectors(ax1, ax2, tax1, tax2)
   # print(tax1)
   # print(x@ax1)
   # print(tax2)
   # print(x@ax2)
   assert np.allclose(x @ ax1, tax1, atol=1e-2)
   assert np.allclose(x @ ax2, tax2, atol=1e-2)

def test_calc_dihedral_angle():
   dang = calc_dihedral_angle(
      [1.0, 0.0, 0.0],
      [0.0, 0.0, 0.0],
      [0.0, 1.0, 0.0],
      [0.0, 0.0, 1.0],
   )
   assert np.allclose(dang, -np.pi / 2)
   dang = calc_dihedral_angle(
      [1.0, 0.0, 0.0],
      [0.0, 0.0, 0.0],
      [0.0, 1.0, 0.0],
      [1.0, 0.0, 1.0],
   )
   assert np.allclose(dang, -np.pi / 4)
   dang = calc_dihedral_angle(
      [1.0, 0.0, 0.0, 1.0],
      [0.0, 0.0, 0.0, 1.0],
      [0.0, 1.0, 0.0, 1.0],
      [1.0, 0.0, 1.0, 1.0],
   )
   assert np.allclose(dang, -np.pi / 4)

def test_align_lines_dof_dihedral_rand_single():
   fix, mov, dof = rand_unit(3)

   if angle(fix, dof) > np.pi / 2: dof = -dof
   if angle(dof, mov) > np.pi / 2: mov = -mov
   target_angle = angle(mov, fix)
   dof_angle = angle(mov, dof)
   fix_to_dof_angle = angle(fix, dof)

   if target_angle + dof_angle < fix_to_dof_angle: return

   axis = hcross(fix, dof)
   mov_in_plane = (hrot(axis, -dof_angle) @ dof[..., None]).reshape(1, 4)
   # could rotate so mov is in plane as close to fix as possible
   # if hdot(mov_in_plane, fix) < 0:
   #    mov_in_plane = (hrot(axis, np.py + dof_angle) @ dof[..., None]).reshape(1, 4)

   test = calc_dihedral_angle(fix, [0.0, 0.0, 0.0, 0.0], dof, mov_in_plane)
   assert np.allclose(test, 0) or np.allclose(test, np.pi)
   dang = calc_dihedral_angle(fix, [0.0, 0.0, 0.0, 0.0], dof, mov)

   ahat = rotation_around_dof_for_target_angle(target_angle, dof_angle, fix_to_dof_angle)
   # print(ahat, dang, abs(dang) + abs(ahat))

   # print('result', 'ta', np.degrees(target_angle), 'da', np.degrees(dof_angle), 'fda',
   # np.degrees(fix_to_dof_angle), dang, ahat, abs(abs(dang) - abs(ahat)))

   atol = 1e-5 if 0.05 < abs(dang) < np.pi - 0.05 else 1e-2
   close1 = np.allclose(abs(dang), abs(ahat), atol=atol)
   close2 = np.allclose(abs(dang), np.pi - abs(ahat), atol=atol)
   if not close1 or close2:
      print('ERROR', abs(dang), abs(ahat), np.pi - abs(ahat))
   assert close1 or close2

def test_align_lines_dof_dihedral_rand_3D():
   num_sol_found, num_total, num_no_sol, max_sol = [0] * 4
   for i in range(100):
      target_angle = np.random.uniform(0, np.pi)
      fix, mov, dof = rand_unit(3)

      if hdot(dof, fix) < 0:
         dof = -dof
      if angle(dof, mov) > np.pi / 2:
         mov = -mov

      if line_angle(fix, dof) > line_angle(mov, dof) + target_angle:
         continue
      if target_angle > line_angle(mov, dof) + line_angle(fix, dof):
         continue

      solutions = xform_around_dof_for_vector_target_angle(fix, mov, dof, target_angle)
      if solutions is None:
         continue

      num_sol_found += 0 < len(solutions)
      max_sol = np.maximum(max_sol, target_angle)
      num_total += 1

      for sol in solutions:
         assert np.allclose(target_angle, angle(fix, sol @ mov), atol=1e-5)

   print(num_total, num_sol_found, num_no_sol, np.degrees(max_sol))
   assert (num_sol_found) / num_total > 0.6

def test_align_lines_dof_dihedral_rand(n=100):
   for i in range(n):
      # print(i)
      test_align_lines_dof_dihedral_rand_single()

def test_align_lines_dof_dihedral_basic():
   target_angle = np.radians(30)
   dof_angle = np.radians(30)
   fix_to_dof_angle = np.radians(60)
   ahat = rotation_around_dof_for_target_angle(target_angle, dof_angle, fix_to_dof_angle)
   assert np.allclose(ahat, 0)

   target_angle = np.radians(30)
   dof_angle = np.radians(30)
   fix_to_dof_angle = np.radians(30)
   ahat = rotation_around_dof_for_target_angle(target_angle, dof_angle, fix_to_dof_angle)
   assert np.allclose(ahat, 1.088176213364169)

   target_angle = np.radians(45)
   dof_angle = np.radians(30)
   fix_to_dof_angle = np.radians(60)
   ahat = rotation_around_dof_for_target_angle(target_angle, dof_angle, fix_to_dof_angle)
   assert np.allclose(ahat, 0.8853828498391183)

def test_place_lines_to_isect_F432():
   ta1 = hnormalized([0., 1., 0., 0.])
   tp1 = np.array([0., 0., 0., 1])
   ta2 = hnormalized([0., -0.5, 0.5, 0.])
   tp2 = np.array([-1, 1, 1, 1.])
   sl2 = hnormalized(tp2 - tp1)

   for i in range(100):
      Xptrb = rand_xform(cart_sd=0)
      ax1 = Xptrb @ np.array([0., 1., 0., 0.])
      pt1 = Xptrb @ np.array([0., 0., 0., 1.])
      ax2 = Xptrb @ np.array([0., -0.5, 0.5, 0.])
      pt2 = Xptrb @ hnormalized(np.array([-1.0, 1.0, 1.0, 1.]))

      Xalign, delta = align_lines_isect_axis2(pt1, ax1, pt2, ax2, ta1, tp1, ta2, sl2)
      xp1, xa1 = Xalign @ pt1, Xalign @ ax1
      xp2, xa2 = Xalign @ pt2, Xalign @ ax2
      assert np.allclose(Xalign[3, 3], 1.0)

      # print('ax1', xa1, ta1)
      # print('ax2', xa2, ta2)
      # print('pt1', xp1)
      # print('pt2', xp2)

      assert np.allclose(line_angle(xa1, xa2), line_angle(ta1, ta2))
      assert np.allclose(line_angle(xa1, ta1), 0.0, atol=0.001)
      assert np.allclose(line_angle(xa2, ta2), 0.0, atol=0.001)
      isect_error = line_line_distance_pa(xp2, xa2, [0, 0, 0, 1], sl2)
      assert np.allclose(isect_error, 0, atol=0.001)

def test_place_lines_to_isect_onecase():
   tp1 = np.array([+0, +0, +0, 1])
   ta1 = np.array([+1, +1, +1, 0])
   ta2 = np.array([+1, +1, -1, 0])
   sl2 = np.array([+0, +1, +1, 0])
   pt1 = np.array([+0, +0, +0, 1])
   ax1 = np.array([+1, +1, +1, 0])
   pt2 = np.array([+1, +2, +1, 1])
   ax2 = np.array([+1, +1, -1, 0])
   ta1 = hnormalized(ta1)
   ta2 = hnormalized(ta2)
   sl2 = hnormalized(sl2)
   ax1 = hnormalized(ax1)
   ax2 = hnormalized(ax2)

   Xalign, delta = align_lines_isect_axis2(pt1, ax1, pt2, ax2, ta1, tp1, ta2, sl2)
   isect_error = line_line_distance_pa(Xalign @ pt2, Xalign @ ax2, [0, 0, 0, 1], sl2)
   assert np.allclose(isect_error, 0, atol=0.001)

def test_place_lines_to_isect_F432_null():
   ta1 = np.array([0., 1., 0., 0.])
   tp1 = np.array([0., 0., 0., 1.])
   ta2 = np.array([0., -0.5, 0.5, 0.])
   tp2 = np.array([-0.57735, 0.57735, 0.57735, 1.])
   sl2 = tp2 - tp1

   ax1 = np.array([0., 1., 0., 0.])
   pt1 = np.array([0., 0., 0., 1.])
   ax2 = np.array([0., -0.5, 0.5, 0.])
   pt2 = np.array([-0.57735, 0.57735, 0.57735, 1.])

   Xalign, delta = align_lines_isect_axis2(pt1, ax1, pt2, ax2, ta1, tp1, ta2, sl2)
   assert np.allclose(Xalign[3, 3], 1.0)

   xp1, xa1 = Xalign @ pt1, Xalign @ ax1
   xp2, xa2 = Xalign @ pt2, Xalign @ ax2
   assert np.allclose(line_angle(xa1, xa2), line_angle(ta1, ta2))
   assert np.allclose(line_angle(xa1, ta1), 0.0)
   assert np.allclose(line_angle(xa2, ta2), 0.0, atol=0.001)
   isect_error = line_line_distance_pa(xp2, xa2, [0, 0, 0, 1], sl2)
   assert np.allclose(isect_error, 0, atol=0.001)

def test_expand_xforms_basic():
   x1 = hrot([0, 0, 1], np.pi)
   x2 = hrot([1, 1, 1], np.pi * 2 / 3, [1, 0, 0])
   xgen = x1, x2
   frames = expand_xforms(xgen)
   points = np.stack([x @ [1, 0, 0, 1] for x in frames])

   correct = np.array([(-1.0, 0.0, 0.0, 1), (1.0, -0.0, 0.0, 1), (-1.0, 0.0, 0.0, 1),
                       (1.0, -2.0, 0.0, 1), (1.0, 0.0, 0.0, 1), (-1.0, 2.0, 0.0, 1),
                       (-1.0, 0.0, 0.0, 1), (1.0, -2.0, 0.0, 1), (1.0, -0.0, -2.0, 1)])
   assert np.allclose(correct, points)

def _vaildate_test_scale_translate_lines_isect_lines(samp, xalign, scale, i):
   assert xalign is not None
   assert scale is not None
   pt1, ax1, pt2, ax2, tp1, ta1, tp2, ta2 = samp
   ok_ax1 = (np.allclose(xalign @ ax1, ta1, atol=1e-5)
             or np.allclose(xalign @ -ax1, ta1, atol=1e-5))
   ok_ax2 = (np.allclose(xalign @ ax2, ta2, atol=1e-5)
             or np.allclose(xalign @ -ax2, ta2, atol=1e-5))

   # ok_pt1 = np.allclose(xalign @ pt1, scale * tp1, atol=1e-3)
   # if not ok_pt1:
   #    offset1 = hm.hnormalized(xalign @ pt1 - scale * tp1)
   #    offset1[3] = 0
   #    ok_pt1 = (np.allclose(offset1, ta1, atol=1e-3) or np.allclose(offset1, -ta1, atol=1e-3))
   # ok_pt2 = np.allclose(xalign @ pt2, scale * tp2, atol=1e-3)
   # if not ok_pt2:
   #    offset2 = hm.hnormalized(xalign @ pt2 - scale * tp2)
   #    offset2[3] = 0
   #    ok_pt2 = (np.allclose(offset2, ta2, atol=1e-3) or np.allclose(offset2, -ta2, atol=1e-3))
   dis1 = np.linalg.norm((xalign @ pt1 - scale * tp1))
   if dis1 > 0.009:
      dis1 = np.sin(angle(xalign @ pt1 - scale * tp1, ta1)) * dis1

   dis2 = np.linalg.norm((xalign @ pt2 - scale * tp2))
   if dis2 > 0.009:
      dis2 = np.sin(angle(xalign @ pt2 - scale * tp2, ta2)) * dis2
   ok_pt1 = abs(dis1) < 0.01  # this is *really* loose to allow very rare cases
   ok_pt2 = abs(dis2) < 0.01  # in 99999/100000 cases, much tighter

   if not (ok_ax1 and ok_ax2 and ok_pt1 and ok_pt2):

      print()
      print('norm', np.linalg.norm((xalign @ pt1 - scale * tp1)),
            np.linalg.norm((xalign @ pt2 - scale * tp2)))
      print('dis', dis1, dis2)
      print('sin', np.sin(angle(xalign @ pt1 - scale * tp1, ta1)),
            np.sin(angle(xalign @ pt2 - scale * tp2, ta2)))
      print()

      print(ta1, np.linalg.norm(ta1))
      print(xalign @ pt1)
      print(scale * tp1)

      print()
      print(np.linalg.norm(xalign @ pt2))
      print(np.linalg.norm(xalign @ pt2 - scale * tp2))
      print(ta2, np.linalg.norm(ta2))
      print(xalign @ pt2)
      print(scale * tp2)

      if not ok_ax1: print('fail ax1 on %i' % i)
      if not ok_ax2: print('fail ax2 on %i' % i)
      if not ok_pt1: print('fail pt1 on %i' % i)
      if not ok_pt2: print('fail pt2 on %i' % i)
      print(repr(samp))

      if 0:
         rays = np.array([
            hm.hray(xalign @ pt1, xalign @ ax1),
            hm.hray(xalign @ pt2, xalign @ ax2),
            hm.hray(scale * tp1, scale * ta1),
            hm.hray(scale * tp2, scale * ta2),
         ])
         colors = [(1, 0, 0), (0, 0, 1), (0.8, 0.5, 0.5), (0.5, 0.5, 0.8)]
         rp.viz.showme(rays, colors=colors, block=False)
      assert ok_ax1 and ok_ax2 and ok_pt1 and ok_pt2

def test_scale_translate_lines_isect_lines_p4132():
   samps = list()
   for i in range(30):
      pt1 = np.array([-40, 0, -40, 1], dtype='d')
      ax1 = np.array([-1, -1, 1, 0], dtype='d')
      pt2 = np.array([-20, -20, -20, 1], dtype='d')
      ax2 = np.array([0, -1, -1, 0], dtype='d')

      tp1 = np.array([-0.5, 0, -0.5, 1], dtype='d')
      ta1 = np.array([-1, -1, 1, 0], dtype='d')
      tp2 = np.array([-0.125, -0.125, -0.125, 1], dtype='d')
      ta2 = np.array([0, -1, -1, 0], dtype='d')

      ax1 = hnormalized(ax1)
      ax2 = hnormalized(ax2)
      ta1 = hnormalized(ta1)
      ta2 = hnormalized(ta2)
      tmp = hm.rand_vec() * 30
      pt1 += tmp
      pt2 += tmp
      pt1 += ax1 * hm.rand_vec() * 30
      pt2 += ax2 * hm.rand_vec() * 30
      xtest = hm.rand_xform()
      pt1 = xtest @ pt1
      ax1 = xtest @ ax1
      pt2 = xtest @ pt2
      ax2 = xtest @ ax2
      samps.append((pt1, ax1, pt2, ax2, tp1, ta1, tp2, ta2))

   # from numpy import array
   # samps = [(array([44.83313235, 54.92005006, -21.4824442,
   #                  1.]), array([-0.40516428, -0.41266195, 0.81581372,
   #                               0.]), array([16.22902529, 0.36515706, 18.20013359, 1.]),
   #           array([-0.87280016, 0.4402107, -0.21079474, 0.]), array([-0.5, 0., -0.5, 1.]),
   #           array([-0.57735027, -0.57735027, 0.57735027, 0.]), array(
   #              [-0.125, -0.125, -0.125, 1.]), array([0., -0.70710678, -0.70710678, 0.]))]

   for i, samp in enumerate(samps):
      xalign, scale = scale_translate_lines_isect_lines(*samp)
      _vaildate_test_scale_translate_lines_isect_lines(samp, xalign, scale, i)

def test_scale_translate_lines_isect_lines_nonorthog():
   nsamp = 30
   samps = list()
   for i in range(nsamp):
      pt1 = np.array([-40, 0, -40, 1], dtype='d')
      ax1 = np.array([1, 1, 1, 0], dtype='d')
      pt2 = np.array([-20, -20, -20, 1], dtype='d')
      ax2 = np.array([0, 1, 1, 0], dtype='d')

      tp1 = np.array([-0.5, 0, -0.5, 1], dtype='d')
      ta1 = np.array([1, 1, 1, 0], dtype='d')
      tp2 = np.array([-0.125, -0.125, -0.125, 1], dtype='d')
      ta2 = np.array([0, 1, 1, 0], dtype='d')

      ax1 = hnormalized(ax1)
      ax2 = hnormalized(ax2)
      ta1 = hnormalized(ta1)
      ta2 = hnormalized(ta2)
      tmp = hm.rand_vec() * 30
      pt1 += tmp
      pt2 += tmp
      pt1 += ax1 * hm.rand_vec() * 30
      pt2 += ax2 * hm.rand_vec() * 30
      xtest = hm.rand_xform()
      pt1 = xtest @ pt1
      ax1 = xtest @ ax1
      pt2 = xtest @ pt2
      ax2 = xtest @ ax2
      samps.append((pt1, ax1, pt2, ax2, tp1, ta1, tp2, ta2))

   # from numpy import array
   # samps = [(array([9.9106826, 32.13286237, -17.17714398,
   #                  1.]), array([-0.75428055, -0.53955273, 0.37409049,
   #                               0.]), array([16.67226665, 9.12240355, -10.38776327, 1.]),
   #           array([-0.23706077, -0.78614464, 0.57077035, 0.]), array([-0.5, 0., -0.5, 1.]),
   #           array([0.57735027, 0.57735027, 0.57735027, 0.]), array(
   #              [-0.125, -0.125, -0.125, 1.]), array([0., 0.70710678, 0.70710678, 0.]))]

   ok = 0
   for i, samp in enumerate(samps):

      xalign, scale = scale_translate_lines_isect_lines(*samp)
      if xalign is not None:
         ok += 1
         _vaildate_test_scale_translate_lines_isect_lines(samp, xalign, scale, i)
      else:
         pass

   # print(ok, nsamp)
   assert ok > nsamp * 0.5

def test_scale_translate_lines_isect_lines_arbitrary():
   samps = list()
   for i in range(30):
      tp1 = hm.rand_point()
      ta1 = hm.rand_unit()
      tp2 = hm.rand_point()
      ta2 = hm.rand_unit()
      rx = hm.rand_xform()
      pt1 = rx @ tp1
      ax1 = rx @ ta1
      pt2 = rx @ tp2
      ax2 = rx @ ta2
      scale = -np.random.rand() * 2
      tp1[:3] = tp1[:3] * scale
      tp2[:3] = tp2[:3] * scale
      if np.random.rand() < 0.8:
         tp1[:3] += np.random.normal() * ta1[:3]
         tp2[:3] += np.random.normal() * ta2[:3]

      # ??? proj_perp(_ta2, _tp2 - _pt2) always 0

      samps.append((pt1, ax1, pt2, ax2, tp1, ta1, tp2, ta2))

   # from numpy import array
   # samps = [(array([9.75754336, 11.94442093, -27.9587031,
   #                  1.]), array([-0.34856845, 0.5450748, 0.76249164,
   #                               0.]), array([35.97908368, -17.0049886, 37.47481606, 1.]),
   #           array([-0.67119914, 0.0327908, 0.74055147, 0.]), array([-0.5, 0., -0.5, 1.]),
   #           array([0.57735027, 0.57735027, 0.57735027, 0.]), array(
   #              [-0.125, -0.125, -0.125, 1.]), array([0., 0.70710678, 0.70710678, 0.]))]

   for i, samp in enumerate(samps):
      xalign, scale = scale_translate_lines_isect_lines(*samp)
      _vaildate_test_scale_translate_lines_isect_lines(samp, xalign, scale, i)

def test_scale_translate_lines_isect_lines_cases():
   from numpy import array, float32
   samps = [
      (
         array([1.37846362, 0.86161002, -0.27543099, 1.]),
         array([-0.00864379, 0.06366346, 0.99793399, 0.]),
         array([1.32846251, 1.93970668, -1.21464696, 1.]),
         array([0.00118571, -0.06660165, -0.99777894, 0.]),
         array([-0.9762876, -0.33263428, -1.85131458, 1.]),
         array([-0.39919588, -0.753445, 0.52245888, 0.]),
         array([-2.21895034, -1.82820754, -2.95265392, 1.]),
         array([0.39191218, 0.75669448, -0.52327651, 0.]),
      ),
      (
         array([0.51437861, 2.18780994, -1.99001668, 1.]),
         array([-0.18190349, 0.31861761, -0.93026552, 0.]),
         array([0.61818801, 0.22765824, -1.97645497, 1.]),
         array([0.71970973, -0.68295276, -0.12487363, 0.]),
         array([-3.90681481, -2.81324474, -1.48884669, 1.]),
         array([-0.5400405, 0.06772283, 0.83890994, 0.]),
         array([-1.99738608, -2.86968414, -1.03712605, 1.]),
         array([0.66836603, -0.71331712, 0.21086854, 0.]),
      ),
      (
         array([0.80453348, -1.70369142, -1.56996154, 1.]),
         array([-0.10405314, 0.57145446, 0.81401028, 0.]),
         array([1.23167311, -2.28086782, -2.31477258, 1.]),
         array([-0.48977051, -0.79128183, -0.36605725, 0.]),
         array([-0.18173768, 0.38484544, 1.38040017, 1.]),
         array([0.73656647, 0.47040127, -0.48599638, 0.]),
         array([-0.99021092, 0.20752128, 2.0010865, 1.]),
         array([-0.70175834, -0.66130471, -0.26497418, 0.]),
      ),
      (
         array([0.66161907, 0.53607942, -0.40837472, 1.]),
         array([0.52195716, 0.4186833, -0.74314535, 0.]),
         array([1.01258715, 0.05252822, -0.08320797, 1.]),
         array([-0.07314979, 0.73456495, 0.6745839, 0.]),
         array([1.70065761, -1.66626863, -0.01729367, 1.]),
         array([-0.24527134, -0.85585473, 0.4553621, 0.]),
         array([1.21956504, -1.18959931, 0.04650348, 1.]),
         array([0.75544244, 0.34941661, 0.55426958, 0.]),
      ),
      (
         array([-0.2624203, 0.88704277, 1.44731444, 1.]),
         array([0.68508642, -0.55063177, -0.47692898, 0.]),
         array([-0.10187965, 1.78492688, 3.99709701, 1.]),
         array([-0.41175151, 0.82175847, 0.39392095, 0.]),
         array([-1.26813184, -0.15104216, -0.70483344, 1.]),
         array([0.7067224, 0.39894059, 0.58428576, 0.]),
         array([-2.62475518, 1.12221865, -2.67250729, 1.]),
         array([-0.3987861, -0.39064141, -0.82968002, 0.]),
      ),
      (
         array([-76.25620827, 46.15603441, 39.92563141, 1.]),
         array([0.97109258, -0.12288913, 0.20463984, 0.]),
         array([-72.71041931, 58.85680507, 17.2434256, 1.]),
         array([0.65593139, -0.33220823, 0.67778441, 0.]),
         array([-0.5, 0., -0.5, 1.]),
         array([0.57735027, 0.57735027, 0.57735027, 0.]),
         array([-0.125, -0.125, -0.125, 1.]),
         array([0., 0.70710678, 0.70710678, 0.]),
      ),
      (
         array([0, 0, 0, 1]),
         array([0., 0., 1., 0.]),
         array([2.784926, -7.0175056, -0.07861771, 1.], dtype=float32),
         array([0.65977187, 0.48100702, 0.57735027, 0.]),
         array([0., 0., 0., 1.]),
         array([0.57735027, 0.57735027, 0.57735027, 0.]),
         array([0., -0.25, 0., 1.]),
         array([0., 0., 1., 0.]),
      ),
   ]
   for i, samp in enumerate(samps):
      # print('SAMP', i)
      xalign, scale = scale_translate_lines_isect_lines(*samp)
      _vaildate_test_scale_translate_lines_isect_lines(samp, xalign, scale, i)

def test_xform_around_dof_for_vector_target_angle():

   fix = np.array([0, 0, 1, 0])
   mov = np.array([-0.48615677, 0.14842946, -0.86117379, 0.])
   dof = np.array([-1.16191467, 0.35474642, -2.05820535, 0.])
   target_angle = 0.6154797086703874
   # should not throw
   solutions = xform_around_dof_for_vector_target_angle(fix, mov, dof, target_angle)
   assert solutions == []

if __name__ == '__main__':

   test_axis_angle_vs_axis_angle_cen_performance()

   # test_xform_around_dof_for_vector_target_angle()

   # test_calc_dihedral_angle()
   # test_align_lines_dof_dihedral_basic()
   # test_align_lines_dof_dihedral_rand(10)
   # test_align_lines_dof_dihedral_rand_3D()
   # test_line_line_closest_points()
   # test_place_lines_to_isect_onecase()
   # test_place_lines_to_isect_F432_null()
   # test_place_lines_to_isect_F432()
   # test_expand_xforms_basic()

   # test_scale_translate_lines_isect_lines_cases()
   # for i in range(10):
   #    test_scale_translate_lines_isect_lines_p4132()
   #    test_scale_translate_lines_isect_lines_nonorthog()
   #    test_scale_translate_lines_isect_lines_arbitrary()

   print('test_homog.py done')
