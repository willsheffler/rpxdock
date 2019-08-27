import numpy as np
import rpxdock.homog as hm
from rpxdock.geom import sym

allowed_twocomp_architectures = """
D22 D32 D42 D52 D62 D72 D82
T32 T33 O32 O42 O43 I32 I52 I53
T32D T23D T33D
O32D O23D O42D O24D O43D O34D
I32D I23D I52D I54D I53D I35D
""".split()

class DockSpec1CompCage:
   def __init__(self, spec):
      assert len(spec) == 2
      assert spec in "T2 T3 O2 O3 O4 I2 I3 I5".split()
      self.spec = spec
      self.sym = spec[0]
      self.num_components = 1
      self.nfold = int(spec[1])
      self.is_dihedral = [False]
      assert self.sym in "TOI"
      self.axis = sym.axes[self.sym][self.nfold]
      self.axis_second = sym.axes_second[self.sym][self.nfold]

      self.orig = hm.align_vector([0, 0, 1], self.axis)
      self.to_neighbor_olig = sym.to_neighbor_olig[self.sym][self.nfold]
      self.orig_second = self.to_neighbor_olig @ self.orig

      cang = hm.angle(self.axis, self.axis_second)
      aang = (np.pi - cang) / 2.0
      self.slide_to_axis_displacement = np.sin(aang) / np.sin(cang)

      self.symframes_ = sym.frames[self.sym]

   def slide_dir(self):
      dirn = self.axis_second - self.axis
      return dirn[:3] / np.linalg.norm(dirn)

   def placements(self, angles):
      place1 = hm.hrot(self.axis, angles, degrees=1) @ self.orig
      place2 = place1 @ hm.hrot([1, 0, 0], 180)  # perp to z
      return np.concatenate([place1, place2])

   def placements_second(self, placements):
      return self.to_neighbor_olig @ placements

   def symframes(self, cellspacing=None, radius=None):
      return self.symframes_

   def place_along_axis(self, pos, slide_dist):
      origshape = pos.shape
      pos = pos.reshape(-1, 4, 4)
      # pos2 = pos2.reshape(-1, 4, 4)

      adis = -slide_dist * self.slide_to_axis_displacement
      newt = self.axis * adis[:, None]
      newpos = pos.copy()
      newpos[:, :3, 3] = newt[:, :3]

      # newt2 = self.to_neighbor_olig @ newt[:, :, None]
      # newdelta = newt2.squeeze() - newt
      # olddelta = pos2[:, :, 3] - pos[:, :, 3]
      # assert np.allclose(newdelta, olddelta)

      return newpos.reshape(origshape)

class DockSpec2CompCage:
   def __init__(self, spec):
      self.spec = spec.upper()
      assert self.spec in allowed_twocomp_architectures
      assert len(self.spec) == 3 or self.spec.endswith('D')
      self.sym = spec[0]
      self.num_components = 2
      self.is_dihedral = [False, self.spec.endswith('D')]
      self.nfold1 = int(spec[1])
      self.nfold2 = int(spec[2])
      self.nfold = np.array([self.nfold1, self.nfold2])
      self.axis1 = sym.axes[self.sym][self.nfold1]
      self.axis2 = sym.axes[self.sym][self.nfold2]
      self.axis2 = sym.axes[self.sym][33] if spec == "T33" else self.axis2
      self.axis = np.array([self.axis1, self.axis2])
      self.axisperp = hm.hcross(self.axis1, self.axis2)
      self.orig1 = hm.align_vector([0, 0, 1], self.axis1)
      self.orig2 = hm.align_vector([0, 0, 1], self.axis2)
      self.orig = np.array([self.orig1, self.orig2])
      self.symframes_ = sym.frames[self.sym]
      self.axis1_second = sym.axes_second[self.sym][self.nfold1]
      self.axis2_second = sym.axes_second[self.sym][self.nfold2]
      self.axis_second = [self.axis1_second, self.axis2_second]
      self.to_neighbor_olig1 = sym.to_neighbor_olig[self.sym][self.nfold1]
      self.to_neighbor_olig2 = sym.to_neighbor_olig[self.sym][self.nfold2]
      self.to_neighbor_olig = np.array([self.to_neighbor_olig1, self.to_neighbor_olig2])
      if spec == "T33":
         self.axis2_second = sym.axes_second[self.sym][33]
         self.to_neighbor_olig2 = sym.to_neighbor_olig[self.sym][33]
      self.compframes = np.array([sym.symframes(self.nfold[i], self.axis[i]) for i in [0, 1]])
      fax1 = hm.hcross(self.axis1, hm.hcross(self.axis1, self.axis2))
      fax2 = hm.hcross(self.axis2, hm.hcross(self.axis2, self.axis1))
      self.xflip = hm.hrot([fax1, fax2], np.pi)

   def __str__(self):
      return f'{self.spec} axis1 {self.axis1[:3]} axis2 {self.axis2[:3]}'

   def slide_dir(self, angles):
      axisdelta = self.axis2 - self.axis1
      axisdelta /= np.linalg.norm(axisdelta)
      dirs = hm.hrot(self.axisperp, angles) @ axisdelta
      return dirs[..., :3]

   def placements1(self, angles, flip=True):
      place1 = hm.hrot(self.axis1, angles, degrees=1) @ self.orig1
      if not flip:
         return place1
      place2 = place1 @ hm.hrot([1, 0, 0], 180)
      return np.concatenate([place1, place2])

   def placements2(self, angles):
      return hm.hrot(self.axis2, angles, degrees=1) @ self.orig2

   def symframes(self, cellspacing=None, radius=None):
      return self.symframes_

   def move_to_canonical_unit(self, pos1, pos2):
      origshape = pos1.shape
      pos1 = pos1.reshape(-1, 4, 4).copy()
      pos2 = pos2.reshape(-1, 4, 4).copy()
      xcen = self.symframes_[:, None] @ pos1[:, :, 3, None]
      xcen += self.symframes_[:, None] @ pos2[:, :, 3, None]
      tgt = self.axis1 + self.axis2
      xdot = np.sum(xcen.squeeze(-1) * tgt, axis=2)
      x = self.symframes_[np.argmax(xdot, axis=0)]
      return (x @ pos1).reshape(origshape), (x @ pos2).reshape(origshape)

   def place_along_axes(self, pos1, pos2):
      origshape = pos1.shape
      pos1 = pos1.reshape(-1, 4, 4).copy()
      pos2 = pos2.reshape(-1, 4, 4).copy()
      offset = pos2[:, :3, 3] - pos1[:, :3, 3]
      cang = hm.angle(self.axis1, self.axis2)
      cdis = np.linalg.norm(offset, axis=1)

      offset_norm = offset / cdis[:, None]
      dot1 = hm.hdot(offset_norm, self.axis1)
      dot2 = hm.hdot(offset_norm, self.axis2)
      axisdelta = hm.hnormalized(self.axis2 - self.axis1)
      aofst = np.sign(dot1 + dot2) * hm.angle(offset_norm, axisdelta)
      aang = (np.pi - cang) / 2.0 - aofst
      bang = (np.pi - cang) / 2.0 + aofst

      adis = cdis * np.sin(aang) / np.sin(cang)
      bdis = cdis * np.sin(bang) / np.sin(cang)
      newt1 = self.axis1[:3] * adis[:, None]
      newt2 = self.axis2[:3] * bdis[:, None]
      newoffset = newt2 - newt1

      swap = hm.hdot(offset, newoffset) < 0
      newt1[swap] *= -1
      newt2[swap] *= -1
      assert np.allclose(newt2 - newt1, offset[:, :3])
      pos1[:, :3, 3] = newt1
      pos2[:, :3, 3] = newt2

      return (pos1.reshape(origshape), pos2.reshape(origshape))

class DockSpec3CompCage:
   def __init__(self, spec):
      assert len(spec) == 4
      assert spec in "O432 I532".split()
      self.spec = spec
      self.sym = spec[0]
      self.num_components = 3
      self.symframes_ = sym.frames[self.sym]
      assert self.sym in "TOI"
      self.is_dihedral = [False] * 3
      self.nfold = np.array([int(spec[1]), int(spec[2]), int(spec[3])], dtype='i')
      self.axis = np.array([sym.axes[self.sym][n] for n in self.nfold])
      self.axisperp = [
         hm.hcross(self.axis[1], self.axis[2]),
         hm.hcross(self.axis[2], self.axis[0]),
         hm.hcross(self.axis[0], self.axis[1])
      ]
      self.orig = [hm.align_vector([0, 0, 1], a) for a in self.axis]
      self.axis_second = [sym.axes_second[self.sym][n] for n in self.nfold]
      self.to_neighbor_olig = [sym.to_neighbor_olig[self.sym][n] for n in self.nfold]
      self.compframes = np.array([sym.symframes(self.nfold[i], self.axis[i]) for i in [0, 1, 2]])
      self.xflip = hm.hrot([
         hm.hcross(self.axis[0], hm.hcross(self.axis[0], self.axis[1])),
         hm.hcross(self.axis[1], hm.hcross(self.axis[1], self.axis[2])),
         hm.hcross(self.axis[2], hm.hcross(self.axis[2], self.axis[0]))
      ], np.pi)

class DockSpecMonomerToCyclic:
   def __init__(self, spec):
      assert len(spec) == 2
      assert spec[0] == "C"
      self.spec = spec
      self.num_components = 1
      self.nfold = int(spec[1])
      assert self.nfold > 1
      self.angle = 2 * np.pi / self.nfold

      self.orig = np.eye(4)
      self.to_neighbor_monomer = hm.hrot([0, 0, 1], self.angle)
      self.orig_second = self.to_neighbor_monomer @ self.orig

      self.symframes_ = [hm.hrot([0, 0, 1], self.angle * i) for i in range(self.nfold)]
      self.tan_half_vertex = np.tan((np.pi - self.angle) / 2)

   def slide_dir(self):
      return [1, 0, 0]

   def placements_second(self, placements):
      return self.to_neighbor_monomer @ placements

   def symframes(self, cellspacing=None, radius=None):
      return self.symframes_

   def place_along_axis(self, pos, slide_dist):
      origshape = pos.shape
      pos = pos.reshape(-1, 4, 4)

      print(self.nfold, self.angle, self.tan_half_vertex)
      if self.nfold == 2:
         dx = slide_dist / 2
         dy = 0
      else:
         dx = slide_dist / 2
         dy = dx * self.tan_half_vertex
         print("delta", slide_dist[0], dx[0], dy[0])

      print(np.mean(dx), np.mean(dy))
      newpos = pos.copy()
      newpos[:, 0, 3] = dx
      newpos[:, 1, 3] = dy
      return newpos.reshape(origshape)

_layer_comp_center_directions = dict(P6_632=([0.86602540378, 0.0, 0], [0.86602540378, 0.5, 0]))

class DockSpec3CompLayer(DockSpec3CompCage):
   def __init__(self, spec):
      spec = spec.upper()
      assert spec.startswith('P')
      self.spec = spec
      self.sym = spec.split('_')[0]
      self.nfolds = list(spec.split('_')[1])
      self.directions = _layer_comp_center_directions[spec]
      self.axis = [np.array([0, 0, 1])] * 3
