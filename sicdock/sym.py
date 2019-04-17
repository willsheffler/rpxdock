from homog.sym import *

tetrahedral_axes[33] = tetrahedral_axes[7]
del tetrahedral_axes[7]

symframes = dict(T=tetrahedral_frames, O=octahedral_frames, I=icosahedral_frames)
symaxes = dict(T=tetrahedral_axes, O=octahedral_axes, I=icosahedral_axes)
