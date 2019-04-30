import numpy as np

# oh god, homog angle_of is wrond for 3x3 rotations...


def axis_angle_of_3x3(rots):
    axis = np.stack(
        (
            rots[..., 2, 1] - rots[..., 1, 2],
            rots[..., 0, 2] - rots[..., 2, 0],
            rots[..., 1, 0] - rots[..., 0, 1],
        ),
        axis=-1,
    )
    four_sin2 = np.sum(axis ** 2, axis=-1)
    sin_angl = np.clip(np.sqrt(four_sin2 / 4), -1, 1)
    tr = np.trace(rots, axis1=-1, axis2=-2)
    cos_angl = np.clip((tr - 1) / 2, -1, 1)
    angl = np.arctan2(sin_angl, cos_angl)
    return axis, angl


def angle_of_3x3(rots):
    return axis_angle_of_3x3(rots)[1]
