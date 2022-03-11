from abc import ABC
import numpy as np

class RigidBodyDOF(ABC):
   """base class for rigid body degrees of freedom"""
   def __init__(self, position):
      self.position = position

class DOfSlideAlong(RigidBodyDOF):
   """slide along an axis"""
   def __init__(self, position=np.eye(4)):
      super().__init__(position)

class DofRotateAround(RigidBodyDOF):
   """rotate around an axis"""
   def __init__(self, position):
      super().__init__(position)
