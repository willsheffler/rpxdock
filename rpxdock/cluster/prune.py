import numpy as np
import rpxdock.homog as hm

from rpxdock.cluster import cookie_cutter

def prune_results_2comp(spec, body1, body2, score, pos, mindis=5):
   order = np.argsort(-score)
   score = score[order]
   pos = pos[0][order], pos[1][order]

   a1, d1, a2, d2, flip = extract_2comp_dofs(spec, pos)
   r1 = (body1.rg_xy() + body1.radius_xy_max()) / 2
   r2 = (body2.rg_xy() + body2.radius_xy_max()) / 2
   dofs = np.stack([a1 * r1, d1, a2 * r2, d2, 1000 * flip], axis=1)

   keep, clustid = cookie_cutter(dofs, mindis)

   return score[keep], (pos[0][keep], pos[1][keep])

def extract_2comp_dofs(spec, pos):
   print("prune_results")

   pos1, pos2 = pos

   rot1a = np.linalg.inv(spec.orig1) @ pos1
   rot1b = np.linalg.inv(spec.orig1 @ hm.hrot([1, 0, 0], 180)) @ pos1
   rot2 = np.linalg.inv(spec.orig2) @ pos2

   axis1a, ang1a = hm.axis_angle_of(rot1a)
   axis1b, ang1b = hm.axis_angle_of(rot1b)
   axis2, ang2 = hm.axis_angle_of(rot2)

   # sort out flips
   aorb = np.abs(axis1a[:, 2]) > 0.999
   aorb |= ang1a < 0.01
   bora = np.abs(axis1b[:, 2]) > 0.999
   bora |= ang1b < 0.01
   assert np.all(bora + aorb)
   assert not np.all(bora * aorb)
   ang1 = np.select([aorb, True], [ang1a, ang1b])

   # print("foo", np.round(np.unique(np.round(ang1, 6)) * 180 / np.pi))
   # print("foo", np.round(np.unique(np.round(ang2, 6)) * 180 / np.pi))

   dist1 = np.linalg.norm(pos1[:, :3, 3], axis=1)
   dist2 = np.linalg.norm(pos2[:, :3, 3], axis=1)

   return ang1, dist1, ang2, dist2, aorb
