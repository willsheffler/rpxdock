from rpxdock.homog import *
import pytest

def test_rand_quat():
   rq = rand_quat((1, 2, 3, 5))
   assert rq.shape == (1, 2, 3, 5, 4)
   assert np.allclose(np.linalg.norm(rq, axis=-1), 1)

def test_quat_mult():
   # from pyquaternion
   assert list(quat_multiply([1, 0, 0, 0], [1, 0, 0, 0])) == [1, 0, 0, 0]
   assert list(quat_multiply([1, 0, 0, 0], [0, 1, 0, 0])) == [0, 1, 0, 0]
   assert list(quat_multiply([1, 0, 0, 0], [0, 0, 1, 0])) == [0, 0, 1, 0]
   assert list(quat_multiply([1, 0, 0, 0], [0, 0, 0, 1])) == [0, 0, 0, 1]
   assert list(quat_multiply([0, 1, 0, 0], [1, 0, 0, 0])) == [0, 1, 0, 0]
   assert list(quat_multiply([0, 1, 0, 0], [0, 1, 0, 0])) == [-1, 0, 0, 0]
   assert list(quat_multiply([0, 1, 0, 0], [0, 0, 1, 0])) == [0, 0, 0, 1]
   assert list(quat_multiply([0, 1, 0, 0], [0, 0, 0, 1])) == [0, 0, -1, 0]
   assert list(quat_multiply([0, 0, 1, 0], [1, 0, 0, 0])) == [0, 0, 1, 0]
   assert list(quat_multiply([0, 0, 1, 0], [0, 1, 0, 0])) == [0, 0, 0, -1]
   assert list(quat_multiply([0, 0, 1, 0], [0, 0, 1, 0])) == [-1, 0, 0, 0]
   assert list(quat_multiply([0, 0, 1, 0], [0, 0, 0, 1])) == [0, 1, 0, 0]
   assert list(quat_multiply([0, 0, 0, 1], [1, 0, 0, 0])) == [0, 0, 0, 1]
   assert list(quat_multiply([0, 0, 0, 1], [0, 1, 0, 0])) == [0, 0, 1, 0]
   assert list(quat_multiply([0, 0, 0, 1], [0, 0, 1, 0])) == [0, -1, 0, 0]
   assert list(quat_multiply([0, 0, 0, 1], [0, 0, 0, 1])) == [-1, 0, 0, 0]

def test_rot_quat_conversion_rand():
   x = rand_xform((5, 6, 7), cart_sd=0)
   assert np.all(is_homog_xform(x))
   q = rot_to_quat(x)
   assert np.all(is_valid_quat_rot(q))
   y = quat_to_xform(q)
   assert np.all(is_homog_xform(y))
   assert x.shape == y.shape
   assert np.allclose(x, y)
   q = rand_quat()
   assert np.all(is_valid_quat_rot(q))
   x = quat_to_xform(q)
   assert np.all(is_homog_xform(x))
   p = rot_to_quat(x)
   assert np.all(is_valid_quat_rot(p))
   assert p.shape == q.shape
   assert np.allclose(p, q)

def test_rot_quat_conversion_cases():
   R22 = np.sqrt(2) / 2
   cases = np.array([[1.00, 0.00, 0.00, 0.00], [0.00, 1.00, 0.00, 0.00], [0.00, 0.00, 1.00, 0.00],
                     [0.00, 0.00, 0.00, 1.00], [+0.5, +0.5, +0.5, +0.5], [+0.5, -0.5, -0.5, -0.5],
                     [+0.5, -0.5, +0.5, +0.5], [+0.5, +0.5, -0.5, -0.5], [+0.5, +0.5, -0.5, +0.5],
                     [+0.5, -0.5, +0.5, -0.5], [+0.5, -0.5, -0.5, +0.5], [+0.5, +0.5, +0.5, -0.5],
                     [+R22, +R22, 0.00, 0.00], [+R22, 0.00, +R22, 0.00], [+R22, 0.00, 0.00, +R22],
                     [0.00, +R22, +R22, 0.00], [0.00, +R22, 0.00, +R22], [0.00, 0.00, +R22, +R22],
                     [+R22, -R22, 0.00, 0.00], [+R22, 0.00, -R22, 0.00], [+R22, 0.00, 0.00, -R22],
                     [0.00, +R22, -R22, 0.00], [0.00, +R22, 0.00, -R22], [0.00, 0.00, +R22,
                                                                          -R22]])
   assert np.all(is_valid_quat_rot(cases))
   x = quat_to_xform(cases)
   assert is_homog_xform(x)
   q = xform_to_quat(x)
   assert np.all(is_valid_quat_rot(q))
   assert np.allclose(cases, q)
