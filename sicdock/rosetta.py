import os
import _pickle

import numpy as np

from pyrosetta import *
import pyrosetta.rosetta as ros

init("-beta -mute all")

def get_pose_cached(fname, pdbdir="."):
    path = os.path.join(pdbdir, fname)
    ppath = path + ".pickle"
    if not os.path.exists(ppath):
        pose = pose_from_file(path)
        with open(ppath, "wb") as out:
            _pickle.dump(pose, out)
            return pose
    with open(ppath, "rb") as inp:
        return _pickle.load(inp)


def numpy_stub_from_rosetta_stub(rosstub):
    npstub = np.zeros((4, 4))
    for i in range(3):
        npstub[..., i, 3] = rosstub.v[i]
        for j in range(3):
            npstub[..., i, j] = rosstub.M(i + 1, j + 1)
    npstub[..., 3, 3] = 1.0
    return npstub


def rosetta_stub_from_numpy_stub(npstub):
    rosstub = ros.core.kinematics.Stub()
    rosstub.M.xx = npstub[0, 0]
    rosstub.M.xy = npstub[0, 1]
    rosstub.M.xz = npstub[0, 2]
    rosstub.M.yx = npstub[1, 0]
    rosstub.M.yy = npstub[1, 1]
    rosstub.M.yz = npstub[1, 2]
    rosstub.M.zx = npstub[2, 0]
    rosstub.M.zy = npstub[2, 1]
    rosstub.M.zz = npstub[2, 2]
    rosstub.v.x = npstub[0, 3]
    rosstub.v.y = npstub[1, 3]
    rosstub.v.z = npstub[2, 3]
    return rosstub


def get_bb_stubs(bbcords, which_resi=None):
    assert bbcords.ndim == 3
    assert bbcords.shape[1] >= 3  # n, ca, c

    stub = np.zeros((bbcords.shape[0], 4, 4))
    stub[:, 3, 3] = 1

    n = bbcords[:, 0, :3]
    ca = bbcords[:, 1, :3]
    c = bbcords[:, 2, :3]
    e1 = n - ca
    # e1 = (c + n) / 2.0 - ca  # from old motif stuff to maintain compatibility
    e1 /= np.linalg.norm(e1, axis=1)[:, None]
    e3 = np.cross(e1, c - ca)
    e3 /= np.linalg.norm(e3, axis=1)[:, None]
    e2 = np.cross(e3, e1)
    stub[:, :3, 0] = e1
    stub[:, :3, 1] = e2
    stub[:, :3, 2] = e3
    # magic numbers from rosetta centroids in some set of pdbs
    avg_centroid_offset = [-0.80571551, -1.60735769, 1.46276045]
    t = stub[:, :3, :3] @ avg_centroid_offset + ca
    stub[:, :3, 3] = t
    assert np.allclose(np.linalg.det(stub), 1)
    return stub



def get_centroids(pose0, which_resi=None):
    pose = pose0.clone()
    pyrosetta.rosetta.core.util.switch_to_residue_type_set(pose, "centroid")
    if which_resi is None:
        which_resi = list(range(1, pose.size() + 1))
    coords = []
    for ir in which_resi:
        r = pose.residue(ir)
        if not r.is_protein():
            raise ValueError("non-protein residue %s at position %i" % (r.name(), ir))
        cen = r.xyz("CEN")
        coords.append([cen.x, cen.y, cen.z, 1])
    return np.stack(coords).astype("f8")


def get_bb_coords(pose, which_resi=None):
    if which_resi is None:
        which_resi = list(range(1, pose.size() + 1))
    coords = []
    for ir in which_resi:
        r = pose.residue(ir)
        if not r.is_protein():
            raise ValueError("non-protein residue %s at position %i" % (r.name(), ir))
        n, ca, c, o = r.xyz("N"), r.xyz("CA"), r.xyz("C"), r.xyz("O")
        cb = r.xyz("CB") if r.has("CB") else r.xyz("CA")
        coords.append(
            np.array(
                [
                    [n.x, n.y, n.z, 1],
                    [ca.x, ca.y, ca.z, 1],
                    [c.x, c.y, c.z, 1],
                    [o.x, o.y, o.z, 1],
                    [cb.x, cb.y, cb.z, 1],
                ]
            )
        )
    return np.stack(coords).astype("f8")


def get_cb_coords(pose, which_resi=None):
    if which_resi is None:
        which_resi = list(range(1, pose.size() + 1))
    cbs = []
    for ir in which_resi:
        r = pose.residue(ir)
        if not r.is_protein():
            raise ValueError("non-protein residue %s at position %i" % (r.name(), ir))
        if r.has("CB"):
            cb = r.xyz("CB")
        else:
            cb = r.xyz("CA")
        cbs.append(np.array([cb.x, cb.y, cb.z, 1]))
    return np.stack(cbs).astype("f8")
