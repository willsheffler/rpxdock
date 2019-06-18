import numpy as np

def is_valid_quat_rot(quat):
   assert quat.shape[-1] == 4
   return np.isclose(1, np.linalg.norm(quat, axis=-1))

def quat_to_upper_half(quat):
   ineg0 = (quat[..., 0] < 0)
   ineg1 = (quat[..., 0] == 0) * (quat[..., 1] < 0)
   ineg2 = (quat[..., 0] == 0) * (quat[..., 1] == 0) * (quat[..., 2] < 0)
   ineg3 = (
      (quat[..., 0] == 0) * (quat[..., 1] == 0) * (quat[..., 2] == 0) * (quat[..., 3] < 0))
   # print(ineg0.shape)
   # print(ineg1.shape)
   # print(ineg2.shape)
   # print(ineg3.shape)
   ineg = ineg0 + ineg1 + ineg2 + ineg3
   quat = quat.copy()
   quat[ineg] = -quat[ineg]
   return quat

def rand_quat(shape=()):
   if isinstance(shape, int): shape = (shape, )
   q = np.random.randn(*shape, 4)
   q /= np.linalg.norm(q, axis=-1)[..., np.newaxis]
   return quat_to_upper_half(q)

def rot_to_quat(xform):
   x = np.asarray(xform)
   t0, t1, t2 = x[..., 0, 0], x[..., 1, 1], x[..., 2, 2]
   tr = t0 + t1 + t2
   quat = np.empty(x.shape[:-2] + (4, ))

   case0 = tr > 0
   S0 = np.sqrt(tr[case0] + 1) * 2
   quat[case0, 0] = 0.25 * S0
   quat[case0, 1] = (x[case0, 2, 1] - x[case0, 1, 2]) / S0
   quat[case0, 2] = (x[case0, 0, 2] - x[case0, 2, 0]) / S0
   quat[case0, 3] = (x[case0, 1, 0] - x[case0, 0, 1]) / S0

   case1 = ~case0 * (t0 >= t1) * (t0 >= t2)
   S1 = np.sqrt(1.0 + x[case1, 0, 0] - x[case1, 1, 1] - x[case1, 2, 2]) * 2
   quat[case1, 0] = (x[case1, 2, 1] - x[case1, 1, 2]) / S1
   quat[case1, 1] = 0.25 * S1
   quat[case1, 2] = (x[case1, 0, 1] + x[case1, 1, 0]) / S1
   quat[case1, 3] = (x[case1, 0, 2] + x[case1, 2, 0]) / S1

   case2 = ~case0 * (t1 > t0) * (t1 >= t2)
   S2 = np.sqrt(1.0 + x[case2, 1, 1] - x[case2, 0, 0] - x[case2, 2, 2]) * 2
   quat[case2, 0] = (x[case2, 0, 2] - x[case2, 2, 0]) / S2
   quat[case2, 1] = (x[case2, 0, 1] + x[case2, 1, 0]) / S2
   quat[case2, 2] = 0.25 * S2
   quat[case2, 3] = (x[case2, 1, 2] + x[case2, 2, 1]) / S2

   case3 = ~case0 * (t2 > t0) * (t2 > t1)
   S3 = np.sqrt(1.0 + x[case3, 2, 2] - x[case3, 0, 0] - x[case3, 1, 1]) * 2
   quat[case3, 0] = (x[case3, 1, 0] - x[case3, 0, 1]) / S3
   quat[case3, 1] = (x[case3, 0, 2] + x[case3, 2, 0]) / S3
   quat[case3, 2] = (x[case3, 1, 2] + x[case3, 2, 1]) / S3
   quat[case3, 3] = 0.25 * S3

   assert (np.sum(case0) + np.sum(case1) + np.sum(case2) + np.sum(case3) == np.prod(
      xform.shape[:-2]))

   return quat_to_upper_half(quat)

xform_to_quat = rot_to_quat

def quat_to_rot(quat, dtype='f8', shape=(3, 3)):
   quat = np.asarray(quat)
   assert quat.shape[-1] == 4
   qr = quat[..., 0]
   qi = quat[..., 1]
   qj = quat[..., 2]
   qk = quat[..., 3]
   outshape = quat.shape[:-1]
   rot = np.zeros(outshape + shape, dtype=dtype)
   rot[..., 0, 0] = 1 - 2 * (qj**2 + qk**2)
   rot[..., 0, 1] = 2 * (qi * qj - qk * qr)
   rot[..., 0, 2] = 2 * (qi * qk + qj * qr)
   rot[..., 1, 0] = 2 * (qi * qj + qk * qr)
   rot[..., 1, 1] = 1 - 2 * (qi**2 + qk**2)
   rot[..., 1, 2] = 2 * (qj * qk - qi * qr)
   rot[..., 2, 0] = 2 * (qi * qk - qj * qr)
   rot[..., 2, 1] = 2 * (qj * qk + qi * qr)
   rot[..., 2, 2] = 1 - 2 * (qi**2 + qj**2)
   return rot

def quat_to_xform(quat, dtype='f8'):
   r = quat_to_rot(quat, dtype, shape=(4, 4))
   r[..., 3, 3] = 1
   return r

def quat_multiply(q, r):
   q, r = np.broadcast_arrays(q, r)
   q0, q1, q2, q3 = np.moveaxis(q, -1, 0)
   r0, r1, r2, r3 = np.moveaxis(r, -1, 0)
   assert np.all(q1 == q[..., 1])
   t = np.empty_like(q)
   t[..., 0] = r0 * q0 - r1 * q1 - r2 * q2 - r3 * q3
   t[..., 1] = r0 * q1 + r1 * q0 - r2 * q3 + r3 * q2
   t[..., 2] = r0 * q2 + r1 * q3 + r2 * q0 - r3 * q1
   t[..., 3] = r0 * q3 - r1 * q2 + r2 * q1 + r3 * q0
   return t

def h_rand_points(shape=(1, )):
   pts = np.ones(shape + (4, ))
   pts[..., 0] = np.random.randn(*shape)
   pts[..., 1] = np.random.randn(*shape)
   pts[..., 2] = np.random.randn(*shape)
   return pts

def guess_is_degrees(angle):
   return np.max(np.abs(angle)) > 2 * np.pi

def is_broadcastable(shp1, shp2):
   for a, b in zip(shp1[::-1], shp2[::-1]):
      if a == 1 or b == 1 or a == b:
         pass
      else:
         return False
   return True

def fast_axis_of(xforms):
   return np.stack(
      (xforms[..., 2, 1] - xforms[..., 1, 2], xforms[..., 0, 2] - xforms[..., 2, 0],
       xforms[..., 1, 0] - xforms[..., 0, 1], np.zeros(xforms.shape[:-2])), axis=-1)

def is_homog_xform(xforms):
   return ((xforms.shape[-2:] == (4, 4))
           and (np.allclose(1, np.linalg.det(xforms[..., :3, :3])))
           and (np.allclose(xforms[..., 3, :], [0, 0, 0, 1])))

def hinv(xforms):
   return np.linalg.inv(xforms)

def axis_angle_of_3x3(rots):
   axis = np.stack((
      rots[..., 2, 1] - rots[..., 1, 2],
      rots[..., 0, 2] - rots[..., 2, 0],
      rots[..., 1, 0] - rots[..., 0, 1],
   ), axis=-1)

   four_sin2 = np.sum(axis**2, axis=-1)
   sin_angl = np.clip(np.sqrt(four_sin2 / 4), -1, 1)
   tr = np.trace(rots, axis1=-1, axis2=-2)
   cos_angl = np.clip((tr - 1) / 2, -1, 1)
   angl = np.arctan2(sin_angl, cos_angl)
   axis = axis / np.linalg.norm(axis, axis=-1)[..., np.newaxis]
   return axis, angl

def angle_of_3x3(rots):
   return axis_angle_of_3x3(rots)[1]

def axis_angle_of(xforms):
   if xforms.shape[-1] == 3:
      return axis_angle_of_3x3(xforms)
   axis = fast_axis_of(xforms)
   four_sin2 = np.sum(axis**2, axis=-1)
   axis = axis / np.linalg.norm(axis, axis=-1)[..., np.newaxis]
   sin_angl = np.clip(np.sqrt(four_sin2 / 4), -1, 1)
   cos_angl = np.clip(np.trace(xforms, axis1=-1, axis2=-2) / 2 - 1, -1, 1)
   # tr = 1 + 2*cos
   # cos = (tr-1)/2
   # tr-1 = 1 + 2*cos
   # cos = tr-2/2 = tr/2-1
   angl = np.arctan2(sin_angl, cos_angl)
   return axis, angl

def angle_of(xforms):
   if xforms.shape[-1] == 3:
      return angle_of_3x3(xforms)
   axis = fast_axis_of(xforms)
   four_sin2 = np.sum(axis**2, axis=-1)
   sin_angl = np.clip(np.sqrt(four_sin2 / 4), -1, 1)
   cos_angl = np.clip(np.trace(xforms, axis1=-1, axis2=-2) / 2 - 1, -1, 1)
   angl = np.arctan2(sin_angl, cos_angl)
   return angl

def rot(axis, angle, degrees='auto', dtype='f8', shape=(3, 3)):
   axis = np.array(axis, dtype=dtype)
   angle = np.array(angle, dtype=dtype)
   if degrees is 'auto': degrees = guess_is_degrees(angle)
   angle = angle * np.pi / 180.0 if degrees else angle
   if axis.shape and angle.shape and not is_broadcastable(axis.shape[:-1], angle.shape):
      raise ValueError('axis and angle not compatible: ' + str(axis.shape) + ' ' +
                       str(angle.shape))
   axis /= np.linalg.norm(axis, axis=-1)[..., np.newaxis]
   a = np.cos(angle / 2.0)
   tmp = axis * -np.sin(angle / 2)[..., np.newaxis]
   b, c, d = tmp[..., 0], tmp[..., 1], tmp[..., 2]
   aa, bb, cc, dd = a * a, b * b, c * c, d * d
   bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d
   outshape = angle.shape if angle.shape else axis.shape[:-1]
   rot3 = np.zeros(outshape + shape, dtype=dtype)
   rot3[..., 0, 0] = aa + bb - cc - dd
   rot3[..., 0, 1] = 2 * (bc + ad)
   rot3[..., 0, 2] = 2 * (bd - ac)
   rot3[..., 1, 0] = 2 * (bc - ad)
   rot3[..., 1, 1] = aa + cc - bb - dd
   rot3[..., 1, 2] = 2 * (cd + ab)
   rot3[..., 2, 0] = 2 * (bd + ac)
   rot3[..., 2, 1] = 2 * (cd - ab)
   rot3[..., 2, 2] = aa + dd - bb - cc
   return rot3

def hrot(axis, angle, center=None, dtype='f8', **args):
   axis = np.array(axis, dtype=dtype)
   angle = np.array(angle, dtype=dtype)
   center = (np.array([0, 0, 0], dtype=dtype)
             if center is None else np.array(center, dtype=dtype))
   r = rot(axis, angle, dtype=dtype, shape=(4, 4), **args)
   x, y, z = center[..., 0], center[..., 1], center[..., 2]
   r[..., 0, 3] = x - r[..., 0, 0] * x - r[..., 0, 1] * y - r[..., 0, 2] * z
   r[..., 1, 3] = y - r[..., 1, 0] * x - r[..., 1, 1] * y - r[..., 1, 2] * z
   r[..., 2, 3] = z - r[..., 2, 0] * x - r[..., 2, 1] * y - r[..., 2, 2] * z
   r[..., 3, 3] = 1
   return r

def hpoint(point):
   point = np.asanyarray(point)
   if point.shape[-1] == 4: return point
   elif point.shape[-1] == 3:
      r = np.ones(point.shape[:-1] + (4, ))
      r[..., :3] = point
      return r
   else:
      raise ValueError('point must len 3 or 4')

def hvec(vec):
   vec = np.asanyarray(vec)
   if vec.shape[-1] == 4: return vec
   elif vec.shape[-1] == 3:
      r = np.zeros(vec.shape[:-1] + (4, ))
      r[..., :3] = vec
      return r
   else:
      raise ValueError('vec must len 3 or 4')

def hray(origin, direction):
   origin = hpoint(origin)
   direction = hnormalized(direction)
   s = np.broadcast(origin, direction).shape
   r = np.empty(s[:-1] + (4, 2))
   r[..., :origin.shape[-1], 0] = origin
   r[..., 3, 0] = 1
   r[..., :, 1] = direction
   return r

def hstub(u, v, w, cen=None):
   u, v, w = hpoint(u), hpoint(v), hpoint(w)
   assert u.shape == v.shape == w.shape
   if not cen: cen = u
   cen = hpoint(cen)
   assert cen.shape == u.shape
   stubs = np.empty(u.shape[:-1] + (4, 4))
   stubs[..., :, 0] = hnormalized(u - v)
   stubs[..., :, 2] = hnormalized(hcross(stubs[..., :, 0], w - v))
   stubs[..., :, 1] = hcross(stubs[..., :, 2], stubs[..., :, 0])
   stubs[..., :, 3] = hpoint(cen[..., :])
   return stubs

def htrans(trans, dtype='f8'):
   trans = np.asanyarray(trans)
   if trans.shape[-1] != 3:
      raise ValueError('trans should be shape (..., 3)')
   tileshape = trans.shape[:-1] + (1, 1)
   t = np.tile(np.identity(4, dtype), tileshape)
   t[..., :trans.shape[-1], 3] = trans
   return t

def hdot(a, b):
   a = np.asanyarray(a)
   b = np.asanyarray(b)
   return np.sum(a[..., :3] * b[..., :3], axis=-1)

def hcross(a, b):
   a = np.asanyarray(a)
   b = np.asanyarray(b)
   c = np.zeros(np.broadcast(a, b).shape, dtype=a.dtype)
   c[..., :3] = np.cross(a[..., :3], b[..., :3])
   return c

def hnorm(a):
   a = np.asanyarray(a)
   return np.sqrt(np.sum(a[..., :3] * a[..., :3], axis=-1))

def hnorm2(a):
   a = np.asanyarray(a)
   return np.sum(a[..., :3] * a[..., :3], axis=-1)

def hnormalized(a):
   a = np.asanyarray(a)
   if (not a.shape and len(a) == 3) or (a.shape and a.shape[-1] == 3):
      a, tmp = np.zeros(a.shape[:-1] + (4, )), a
      a[..., :3] = tmp
   return a / hnorm(a)[..., None]

def is_valid_rays(r):
   r = np.asanyarray(r)
   if r.shape[-2:] != (4, 2): return False
   if np.any(r[..., 3, :] != (1, 0)): return False
   if np.any(abs(np.linalg.norm(r[..., :3, 1], axis=-1) - 1) > 0.000001):
      return False
   return True

def rand_point(shape=()):
   if isinstance(shape, int): shape = (shape, )
   return hpoint(np.random.randn(*(shape + (3, ))))

def rand_vec(shape=()):
   if isinstance(shape, int): shape = (shape, )
   return hvec(np.random.randn(*(shape + (3, ))))

def rand_unit(shape=()):
   if isinstance(shape, int): shape = (shape, )
   return hnormalized(np.random.randn(*(shape + (3, ))))

def angle(u, v):
   d = hdot(hnormalized(u), hnormalized(v))
   # todo: handle special cases... 1,-1
   return np.arccos(np.clip(d, -1, 1))

def angle_degrees(u, v):
   return angle(u, v) * 180 / np.pi

def line_angle(u, v):
   a = angle(u, v)
   return np.minimum(a, np.pi - a)

def line_angle_degrees(u, v):
   a = angle(u, v)
   a = np.minimum(a, np.pi - a)
   return a * 180 / np.pi

def rand_ray(shape=(), cen=(0, 0, 0), sdev=1):
   if isinstance(shape, int): shape = (shape, )
   cen = np.asanyarray(cen)
   if cen.shape[-1] not in (3, 4):
      raise ValueError('cen must be len 3 or 4')
   shape = shape or cen.shape[:-1]
   cen = cen + np.random.randn(*(shape + (3, ))) * sdev
   norm = np.random.randn(*(shape + (3, )))
   norm /= np.linalg.norm(norm, axis=-1)[..., np.newaxis]
   r = np.zeros(shape + (4, 2))
   r[..., :3, 0] = cen
   r[..., 3, 0] = 1
   r[..., :3, 1] = norm
   return r

def rand_xform_aac(shape=(), axis=None, ang=None, cen=None):
   if isinstance(shape, int): shape = (shape, )
   if axis is None:
      axis = rand_unit(shape)
   if ang is None:
      ang = np.random.rand(*shape) * np.pi  # todo: make uniform!
   if cen is None:
      cen = rand_point(shape)
   q = rand_quat(shape)
   return hrot(axis, ang, cen)

def rand_xform(shape=(), cart_cen=0, cart_sd=1):
   if isinstance(shape, int): shape = (shape, )
   q = rand_quat(shape)
   x = quat_to_xform(q)
   x[..., :3, 3] = np.random.randn(*shape, 3) * cart_sd + cart_cen
   return x

def proj_perp(u, v):
   u = np.asanyarray(u)
   v = np.asanyarray(v)
   return v - hdot(u, v)[..., None] / hnorm2(u)[..., None] * u

def point_in_plane(plane, pt):
   return np.abs(hdot(plane[..., :3, 1], pt[..., :3] - plane[..., :3, 0])) < 0.000001

def ray_in_plane(plane, ray):
   assert ray.shape[-2:] == (4, 2)
   return (point_in_plane(plane, ray[..., :3, 0]) * point_in_plane(
      plane, ray[..., :3, 0] + ray[..., :3, 1]))

def intersect_planes(plane1, plane2):
   """intersect_Planes: find the 3D intersection of two planes
       Input:  two planes represented by rays shape=(..., 4, 2)
       Output: *L = the intersection line (when it exists)
       Return: rays shape=(...,4,2), status
               0 = intersection returned
               1 = disjoint (no intersection)
               2 = the two planes coincide
    """
   if not is_valid_rays(plane1): raise ValueError('invalid plane1')
   if not is_valid_rays(plane2): raise ValueError('invalid plane2')
   shape1, shape2 = np.array(plane1.shape), np.array(plane2.shape)
   if np.any((shape1 != shape2) * (shape1 != 1) * (shape2 != 1)):
      raise ValueError('incompatible shapes for plane1, plane2:')
   p1, n1 = plane1[..., :3, 0], plane1[..., :3, 1]
   p2, n2 = plane2[..., :3, 0], plane2[..., :3, 1]
   shape = tuple(np.maximum(plane1.shape, plane2.shape))
   u = np.cross(n1, n2)
   abs_u = np.abs(u)
   planes_parallel = np.sum(abs_u, axis=-1) < 0.000001
   p2_in_plane1 = point_in_plane(plane1, p2)
   status = np.zeros(shape[:-2])
   status[planes_parallel] = 1
   status[planes_parallel * p2_in_plane1] = 2
   d1 = -hdot(n1, p1)
   d2 = -hdot(n2, p2)
   amax = np.argmax(abs_u, axis=-1)
   sel0, sel1, sel2 = amax == 0, amax == 1, amax == 2
   n1a, n2a, d1a, d2a, ua = (x[sel0] for x in (n1, n2, d1, d2, u))
   n1b, n2b, d1b, d2b, ub = (x[sel1] for x in (n1, n2, d1, d2, u))
   n1c, n2c, d1c, d2c, uc = (x[sel2] for x in (n1, n2, d1, d2, u))

   ay = (d2a * n1a[..., 2] - d1a * n2a[..., 2]) / ua[..., 0]
   az = (d1a * n2a[..., 1] - d2a * n1a[..., 1]) / ua[..., 0]
   bz = (d2b * n1b[..., 0] - d1b * n2b[..., 0]) / ub[..., 1]
   bx = (d1b * n2b[..., 2] - d2b * n1b[..., 2]) / ub[..., 1]
   cx = (d2c * n1c[..., 1] - d1c * n2c[..., 1]) / uc[..., 2]
   cy = (d1c * n2c[..., 0] - d2c * n1c[..., 0]) / uc[..., 2]
   isect_pt = np.empty(shape[:-2] + (3, ), dtype=plane1.dtype)
   isect_pt[sel0, 0] = 0
   isect_pt[sel0, 1] = ay
   isect_pt[sel0, 2] = az
   isect_pt[sel1, 0] = bx
   isect_pt[sel1, 1] = 0
   isect_pt[sel1, 2] = bz
   isect_pt[sel2, 0] = cx
   isect_pt[sel2, 1] = cy
   isect_pt[sel2, 2] = 0
   isect = hray(isect_pt, u)
   return isect, status

def axis_ang_cen_of_eig(xforms, debug=False):
   raise NotImplemented('this is a bad way to get rotation axis')
   axis, angle = axis_angle_of(xforms)
   # # seems to numerically unstable
   ev, cen = np.linalg.eig(xforms)
   # print(axis)
   # print(cen[..., 0])
   # print(cen[..., 1])
   # print(cen[..., 2])
   # axis = np.real(cen[..., 2])
   cen = np.real(cen[..., 3])
   cen /= cen[..., 3][..., None]
   # # todo: this is unstable.... fix?
   # cen = proj_perp(axis, cen)  # move to reasonable position
   return axis, angle, cen

def axis_ang_cen_of_planes(xforms, debug=False):
   axis, angle = axis_angle_of(xforms)
   #  sketchy magic points...
   p1 = (-32.09501046777237, 03.36227004372687, 35.34672781477340, 1)
   p2 = (21.15113978202345, 12.55664537217840, -37.48294301885574, 1)
   # p1 = rand_point()
   # p2 = rand_point()
   tparallel = hdot(axis, xforms[..., :, 3])[..., None] * axis
   q1 = xforms @ p1 - tparallel
   q2 = xforms @ p2 - tparallel
   n1 = hnormalized(q1 - p1)
   n2 = hnormalized(q2 - p2)
   c1 = (p1 + q1) / 2.0
   c2 = (p2 + q2) / 2.0
   plane1 = hray(c1, n1)
   plane2 = hray(c2, n2)
   isect, status = intersect_planes(plane1, plane2)
   return axis, angle, isect[..., :, 0]

axis_ang_cen_of = axis_ang_cen_of_planes

def line_line_distance_pa(pt1, ax1, pt2, ax2):
   # point1, point2 = hpoint(point1), hpoint(point2)
   # axis1, axis2 = hnormalized(axis1), hnormalized(axis2)
   n = abs(hdot(pt2 - pt1, hcross(ax1, ax2)))
   d = hnorm(hcross(ax1, ax2))
   r = np.zeros_like(n)
   i = abs(d) > 0.00001
   r[i] = n[i] / d[i]
   pp = hnorm(proj_perp(ax1, pt2 - pt1))
   return np.where(np.abs(hdot(ax1, ax2)) > 0.9999, pp, r)

def line_line_distance(ray1, ray2):
   pt1, pt2 = ray1[..., :, 0], ray2[..., :, 0]
   ax1, ax2 = ray1[..., :, 1], ray2[..., :, 1]
   return line_line_distance_pa(pt1, ax1, pt2, ax2)

def line_line_closest_points_pa(pt1, ax1, pt2, ax2, verbose=0):
   C21 = pt2 - pt1
   M = hcross(ax1, ax2)
   m2 = np.sum(M**2, axis=-1)[..., None]
   R = hcross(C21, M / m2)
   t1 = hdot(R, ax2)[..., None]
   t2 = hdot(R, ax1)[..., None]
   Q1 = pt1 - t1 * ax1
   Q2 = pt2 - t2 * ax2
   if verbose:
      print('C21', C21)
      print('M', M)
      print('m2', m2)
      print('R', R)
      print('t1', t1)
      print('t2', t2)
      print('Q1', Q1)
      print('Q2', Q2)
   return Q1, Q2

def line_line_closest_points(ray1, ray2, verbose=0):
   "currently errors if ax1==ax2"
   # pt1, pt2 = hpoint(pt1), hpoint(pt2)
   # ax1, ax2 = hnormalized(ax1), hnormalized(ax2)
   pt1, pt2 = ray1[..., :, 0], ray2[..., :, 0]
   ax1, ax2 = ray1[..., :, 1], ray2[..., :, 1]
   return line_line_closest_points_pa(pt1, ax1, pt2, ax2)

def dihedral(p1, p2, p3, p4):
   p1, p2, p3, p4 = hpoint(p1), hpoint(p2), hpoint(p3), hpoint(p4)
   a = hnormalized(p2 - p1)
   b = hnormalized(p3 - p2)
   c = hnormalized(p4 - p3)
   x = np.clip(hdot(a, b) * hdot(b, c) - hdot(a, c), -1, 1)
   y = np.clip(hdot(a, hcross(b, c)), -1, 1)
   return np.arctan2(y, x)

def align_around_axis(axis, u, v):
   return hrot(axis, -dihedral(u, axis, [0, 0, 0, 0], v))

def align_vector(a, b):
   return hrot((hnormalized(a) + hnormalized(b)) / 2, np.pi)

def align_vectors(a1, a2, b1, b2):
   "minimizes angular error"
   a1, a2, b1, b2 = (hnormalized(v) for v in (a1, a2, b1, b2))
   aaxis = (a1 + a2) / 2.0
   baxis = (b1 + b2) / 2.0
   Xmiddle = align_vector(aaxis, baxis)
   Xaround = align_around_axis(baxis, Xmiddle @ a1, b1)
   X = Xaround @ Xmiddle
   assert (angle(b1, a1) + angle(b2, a2)) + 0.001 >= (angle(b1, X @ a1) + angle(b2, X @ a2))
   return X
