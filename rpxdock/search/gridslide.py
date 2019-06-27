import itertools as it
import numpy as np
import rpxdock.sampling.orientations as ori
import rpxdock.homog as hm

def samples_1xMonomer_orientations(resl):
   quats = ori.quaternion_set_with_covering_radius_degrees(resl)[0]
   return hm.quat_to_xform(quats)

def _tweak_resl():
   npts = np.ceil(360 / spec.nfold / resl)
   resl = 360 / spec.nfold

def samples_1xCyclic(spec, resl=1):
   return spec.placements(np.arange(0, 360 // spec.nfold + 0.9 * resl, resl))

def samples_2xCyclic_slide(spec, resl=1, max_out_of_plane_angle=10, **kw):
   tip = np.ceil(max_out_of_plane_angle / resl) * resl + 0.01
   rots1 = np.arange(0, 360 // spec.nfold1, resl)
   rots2 = np.arange(0, 360 // spec.nfold2, resl)
   # print("rots1", rots1)
   # print("rots2", rots2)
   rots1 = spec.placements1(rots1)
   rots2 = spec.placements2(rots2)
   slideposdn = np.arange(0, -0.001 - tip, -resl)[::-1]
   slideposup = np.arange(resl, tip + 0.001, resl)
   slidenegdn = np.arange(180, 179.999 - tip, -resl)[::-1]
   slidenegup = np.arange(180 + resl, tip + 180.001, resl)
   slides = np.concatenate([slideposdn, slideposup, slidenegdn, slidenegup])
   # print("slides", slides)
   slides = spec.slide_dir(slides)
   return rots1, rots2, slides

def find_connected_1xCyclic_slide(spec, body, samples, min_contacts=30, contact_dis=8.0):
   body2 = body.copy()  # shallow copy except pos
   samples_second = spec.placements_second(samples)
   maxsize = len(samples)
   npair = np.empty(maxsize, np.int32)
   pos = np.empty((maxsize, 4, 4))
   dslide = np.empty(maxsize)
   nresult, nhit = 0, 0
   dirn = spec.slide_dir()
   for x1, x2 in zip(samples, samples_second):
      body.move_to(x1)
      body2.move_to(x2)
      d = body.slide_to(body2, dirn)
      if d < 9e8:
         nhit += 1
         npair0 = body.contact_count(body2, contact_dis)
         if npair0 >= min_contacts:
            npair[nresult] = npair0
            pos[nresult] = body.pos
            dslide[nresult] = d
            nresult += 1
   assert nhit == maxsize
   pos = spec.place_along_axis(pos[:nresult], dslide[:nresult])
   return npair[:nresult], pos

def _check_1body_contacts(body, pos, x_to_neighbor_olig, contact_dis):
   npair = np.zeros(len(pos), "i4") - 1
   body_b = body.copy()
   for i, pos in enumerate(pos):
      body.move_to(pos)
      body_b.move_to(x_to_neighbor_olig @ pos)
      if not body.intersect(body_b):
         npair[i] = body.contact_count(body_b, contact_dis)
   return npair

def find_connected_2xCyclic_slide(spec, body1, body2, samples, min_contacts=30, contact_dis=8.0,
                                  onebody=True, **kw):
   if len(np.unique(samples[2], axis=0)) == len(samples[2]):
      maxsize = len(samples[0]) * len(samples[1]) * len(samples[2])
      samples = it.product(*samples)
   else:  # must be sample list if some duplicates (maybe sketchy??)
      assert len(samples[0]) == len(samples[1]) == len(samples[2])
      maxsize = len(samples[0])
      samples = zip(*samples)

   npair = np.empty(maxsize, np.int32)
   pos1 = np.empty((maxsize, 4, 4))
   pos2 = np.empty((maxsize, 4, 4))
   nresult, nhit = 0, 0
   for x1, x2, dirn in samples:
      body1.move_to(x1)
      body2.move_to(x2)
      assert np.allclose(body1.pos[:3, 3], 0)
      assert np.allclose(body2.pos[:3, 3], 0)
      d = body1.slide_to(body2, dirn)
      if d < 9e8:
         nhit += 1
         npair0 = body1.contact_count(body2, contact_dis)
         if npair0 >= min_contacts:
            npair[nresult] = npair0
            pos1[nresult] = body1.pos
            pos2[nresult] = body2.pos
            nresult += 1

   assert nhit == maxsize
   npair3 = np.empty((nresult, 3))
   pos1, pos2 = spec.place_along_axes(pos1[:nresult], pos2[:nresult])
   if not onebody:
      return npair[:nresult], (pos1, pos2)
   npair1 = _check_1body_contacts(body1, pos1, spec.to_neighbor_olig1, contact_dis)
   npair2 = _check_1body_contacts(body2, pos2, spec.to_neighbor_olig2, contact_dis)
   # print("nresult", nresult, np.sum(npair1 >= 0), np.sum(npair2 >= 0))
   ok = (npair1 >= 0) * (npair2 >= 0)
   npair3 = np.stack([npair[:nresult][ok], npair1[ok], npair2[ok]], axis=1)
   return npair3, (pos1[ok], pos2[ok])

def find_connected_monomer_to_cyclic_slide(spec, body, samples, min_contacts, contact_dis):
   body2 = body.copy()  # shallow copy except pos
   samples_second = spec.placements_second(samples)
   maxsize = len(samples)
   npair = np.empty(maxsize, np.int32)
   pos = np.empty((maxsize, 4, 4))
   dslide = np.empty(maxsize)
   nresult, nhit = 0, 0
   dirn = spec.slide_dir()
   for x1, x2 in zip(samples, samples_second):
      body.move_to(x1)
      body2.move_to(x2)
      d = body.slide_to(body2, dirn)
      if d < 9e8:
         nhit += 1
         npair0 = body.contact_count(body2, contact_dis)
         if npair0 >= min_contacts:
            npair[nresult] = npair0
            pos[nresult] = body.pos
            dslide[nresult] = d
            nresult += 1
   assert nhit == maxsize
   pos = spec.place_along_axis(pos[:nresult], dslide[:nresult])
   return npair[:nresult], pos
