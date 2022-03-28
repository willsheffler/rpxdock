# from abc import ABC
# import numpy as np

# class RigidBodyDOF(ABC):
#    """base class for rigid body degrees of freedom"""
#    def __init__(self, position):
#       self.position = position

# class DofSlideRotate(RigidBodyDOF):
#    """rotate around an axis"""
#    def __init__(self, **kw):
#       super().__init__(**kw)

# class DofFull(RigidBodyDOF):
#    """rotate around an axis"""
#    def __init__(self, **kw):
#       super().__init__(**kw)

# # class DOfSlideAlong(RigidBodyDOF):
# #    """slide along an axis"""
# #    def __init__(self, **kw=np.eye(4)):
# #       super().__init__(**kw)
# #
# # class DofRotateAround(RigidBodyDOF):
# #    """rotate around an axis"""
# #    def __init__(self, **kw):
# #       super().__init__(**kw)
