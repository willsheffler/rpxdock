import pytest, numpy as np
from rpxdock.homog import *
import rpxdock.homog as hm
from rpxdock.geom import sym

def test_sym():
   assert sym.tetrahedral_frames.shape == (12, 4, 4)
   assert sym.octahedral_frames.shape == (24, 4, 4)
   assert sym.icosahedral_frames.shape == (60, 4, 4)
   x = np.concatenate([sym.tetrahedral_frames, sym.octahedral_frames, sym.icosahedral_frames])
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
   assert 1e-5 > abs(ax[0] - ax[1])
   assert 1e-5 > abs(ax[2])
   ax, an = axis_angle_of(hrot([0, 1, 0], np.pi))
   assert 1e-5 > abs(ax[0])
   assert 1e-5 > abs(ax[1]) - 1
   assert 1e-5 > abs(ax[2])

   ax, an = axis_angle_of(hrot([0, 1, 0], np.pi * 0.25))
   print(ax, an)
   assert np.allclose(ax, [0, 1, 0, 0], atol=1e-5)
   assert 1e-5 > abs(an - np.pi * 0.25)
   ax, an = axis_angle_of(hrot([0, 1, 0], np.pi * 0.75))
   print(ax, an)
   assert np.allclose(ax, [0, 1, 0, 0], atol=1e-5)
   assert 1e-5 > abs(an - np.pi * 0.75)

   ax, an = axis_angle_of(hrot([1, 0, 0], np.pi / 2))
   print(np.pi / an)
   assert 1e-5 > abs(an - np.pi / 2)

def test_axis_angle_of_rand():
   shape = (
      4,
      5,
      6,
      7,
      8,
   )
   axis = hnormalized(np.random.randn(*shape, 3))
   angl = np.random.random(shape) * np.pi / 2
   rot = hrot(axis, angl, dtype='f8')
   ax, an = axis_angle_of(rot)
   assert np.allclose(axis, ax, rtol=1e-5)
   assert np.allclose(angl, an, rtol=1e-5)

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
   assert np.allclose(lldist0, lldist1, atol=1e-2, rtol=1e-2)  # loose, but rarely fails otherwise

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
   close1 = np.allclose(abs(dang), abs(ahat), atol=1e-5)
   close2 = np.allclose(abs(dang), np.pi - abs(ahat), atol=1e-5)
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

def align_lines_slide_second(pt1, ax1, pt2, ax2, ta1, tp1, ta2, sl2):
   ## make sure to align with smaller axis choice
   if hm.angle(ax1, ax2) > np.pi / 2:
      ax2 = -ax2
   if hm.angle(ta1, ta2) > np.pi / 2:
      ta2 = -ta2
   assert np.allclose(angle(ta1, ta2), angle(ax1, ax2))
   if abs(hm.angle(ta1, ta2)) < 0.1:
      assert 0
      # vector delta between pt2 and pt1
      d = hm.proj_perp(ax1, pt2 - pt1)
      Xalign = hm.align_vectors(ax1, d, ta1, sl2)  # align d to Y axis
      Xalign[..., :, 3] = -Xalign @ pt1
      cell_dist = (Xalign @ pt2)[..., 1]
   else:
      try:
         Xalign = hm.align_vectors(ax1, ax2, ta1, ta2)
         # print(Xalign @ ax1, ta1)
         # assert np.allclose(Xalign @ ax1, ta1, atol=0.0001)
         # assert np.allclose(Xalign @ ax2, ta2, atol=0.0001)
         # print(Xalign)
      except AssertionError as e:
         print("align_vectors error")
         print("   ", ax1)
         print("   ", ax2)
         print("   ", ta1)
         print("   ", ta2)
         raise e
      Xalign[..., :, 3] = -Xalign @ pt1  ## move pt1 to origin
      Xalign[..., 3, 3] = 1
      cen2_0 = Xalign @ pt2  # moving pt2 by Xalign
      D = np.stack([ta1[:3], sl2[:3], ta2[:3]]).T
      A1offset, cell_dist, _ = np.linalg.inv(D) @ cen2_0[:3]
      # print(A1offset, cell_dist)
      Xalign[..., :, 3] = Xalign[..., :, 3] - (A1offset * ta1)

   return Xalign

def test_place_lines_to_isect_F432():
   ta1 = hnormalized([0., 1., 0., 0.])
   tp1 = np.array([0., 0., 0., 1])
   ta2 = hnormalized([0., -0.5, 0.5, 0.])
   tp2 = np.array([-1, 1, 1, 1.])
   sl2 = hnormalized(tp2 - tp1)

   for i in range(100):
      Xptrb = rand_xform(cart_sd=0)
      # Xptrb = hrot([1, 2, 3], 1.2224)
      ax1 = Xptrb @ np.array([0., 1., 0., 0.])
      pt1 = Xptrb @ np.array([0., 0., 0., 1.])
      ax2 = Xptrb @ np.array([0., -0.5, 0.5, 0.])
      pt2 = Xptrb @ hnormalized(np.array([-1.0, 1.0, 1.0, 1.]))

      Xalign = align_lines_slide_second(pt1, ax1, pt2, ax2, ta1, tp1, ta2, sl2)
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

   Xalign = align_lines_slide_second(pt1, ax1, pt2, ax2, ta1, tp1, ta2, sl2)
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

   Xalign = align_lines_slide_second(pt1, ax1, pt2, ax2, ta1, tp1, ta2, sl2)
   assert np.allclose(Xalign[3, 3], 1.0)

   xp1, xa1 = Xalign @ pt1, Xalign @ ax1
   xp2, xa2 = Xalign @ pt2, Xalign @ ax2
   assert np.allclose(line_angle(xa1, xa2), line_angle(ta1, ta2))
   assert np.allclose(line_angle(xa1, ta1), 0.0)
   assert np.allclose(line_angle(xa2, ta2), 0.0, atol=0.001)
   isect_error = line_line_distance_pa(xp2, xa2, [0, 0, 0, 1], sl2)
   assert np.allclose(isect_error, 0, atol=0.001)

if __name__ == '__main__':
   # test_calc_dihedral_angle()
   # test_align_lines_dof_dihedral_basic()
   # test_align_lines_dof_dihedral_rand(10)
   # test_align_lines_dof_dihedral_rand_3D()
   # test_line_line_closest_points()
   test_place_lines_to_isect_onecase()
   test_place_lines_to_isect_F432_null()
   test_place_lines_to_isect_F432()