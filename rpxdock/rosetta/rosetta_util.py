import numpy as np
import rpxdock as rp
from rpxdock.filter.sscount import secondary_structure_map
from rpxdock.rosetta.triggers_init import rosetta as ros

def rosetta_init(opts="-beta -mute all"):
   from pyrosetta import init
   init(opts)

def numpy_stub_from_rosetta_stub(rosstub):
   npstub = np.zeros((4, 4))
   for i in range(3):
      npstub[..., i, 3] = rosstub.v[i]
      for j in range(3):
         npstub[..., i, j] = rosstub.M(i + 1, j + 1)
   npstub[..., 3, 3] = 1.0
   return npstub

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
         np.array([
            [n.x, n.y, n.z, 1],
            [ca.x, ca.y, ca.z, 1],
            [c.x, c.y, c.z, 1],
            [o.x, o.y, o.z, 1],
            [cb.x, cb.y, cb.z, 1],
         ]))
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

def get_sc_coords(pose, which_resi=None):
   if which_resi is None:
      which_resi = list(range(1, pose.size() + 1))
   resaname, resacrd = list(), list()
   for ir in which_resi:
      r = pose.residue(ir)
      if not r.is_protein():
         raise ValueError("non-protein residue %s at position %i" % (r.name(), ir))
      anames, crd = list(), list()
      for ia in range(r.natoms()):
         anames.append(r.atom_name(ia + 1))
         xyz = r.xyz(ia + 1)
         crd.append([xyz.x, xyz.y, xyz.z])
      resaname.append(anames)
      hcrd = np.ones((len(anames), 4), dtype='f4')
      hcrd[:, :3] = np.array(crd)
      resacrd.append(hcrd)
   return resaname, resacrd

def get_trimmed_poses(
   pose,
   debug=False,
   helix_trim_max=3,
   helix_trim_nres_ignore_end=0,
   individual_pieces=False,
   trim_from_N=True,
   trim_from_C=True,
   **_,
):
   if helix_trim_max == 0 or helix_trim_max is None:
      return [], []
   # print('get_trimmed_poses: ss', pose.secstruct())
   dummybody = rp.Bunch(ss=list(pose.secstruct()))
   ssmap = rp.filter.sscount.secondary_structure_map()
   ssmap.map_body_ss(dummybody)
   # print(ssmap.ss_index)
   # print(ssmap.ss_type_assignments)
   # print(ssmap.ss_element_start)
   # print(ssmap.ss_element_end)
   ssid = np.array(ssmap.ss_index)
   sstype = np.array(ssmap.ss_type_assignments)
   selection = sstype == 'H'
   ssid = ssid[selection]
   sstype = sstype[selection]
   lb = np.array(ssmap.ss_element_start)[selection]
   ub = np.array(ssmap.ss_element_end)[selection] + 1
   # TODO direction matters here !!!
   # print('get_trimmed_poss', lb)
   # print('get_trimmed_poss', ub)
   # print()

   # trim_place_nc = 197  # len(body) - max_trim
   # trim_place_cn = 100  # max_trim
   lb_nc = lb.copy()
   ub_nc = ub.copy()
   lb_nc[0] = 0
   lb_nc[1:] = ub[:-1]

   #  # # print(ssid)
   # print('lb_nc', lb_nc)
   # print('ub_nc', ub_nc)
   # nhelix = np.sum(lb_nc >= trim_place_nc)
   # print(nhelix)
   # print(lb_nc[-nhelix:])
   # print(ub_nc[-nhelix:])
   # print()
   # assert nhelix == 6

   lb_cn = lb.copy()
   ub_cn = ub.copy()
   # print(len(body.ss))
   ub_cn[-1] = len(dummybody.ss)
   ub_cn[:-1] = lb_cn[1:]
   # # lb_cn[0] = 0
   # # lb_cn[1:] = ub[:-1]
   # # print(ssid)
   # print('lb_cn', lb_cn)
   # print('ub_cn', ub_cn)
   # nhelix = np.sum(ub_cn <= trim_place_cn)
   # print(nhelix)
   # print(lb_cn[:nhelix])
   # print(ub_cn[:nhelix])
   # print()
   # print()
   # assert nhelix == 5

   htnie = helix_trim_nres_ignore_end
   # if not individual_pieces:
   # assert helix_trim_nres_ignore_end == 0

   trimN_subbodies = list()
   trimC_subbodies = list()

   if trim_from_N:
      hmtnc = min(helix_trim_max, len(ub_nc) - 1)
      p = ros.core.pose.Pose()
      ros.core.pose.append_subpose_to_pose(p, pose, 1, ub_nc[-hmtnc - 1])
      trimC_subbodies.append(p.clone())
      for i, (start, end) in enumerate(zip(lb_nc[-hmtnc:], ub_nc[-hmtnc:])):
         p = ros.core.pose.Pose()
         begin = start + 1 if individual_pieces else 1
         ros.core.pose.append_subpose_to_pose(p, pose, begin, end - htnie)
         trimC_subbodies.append(p.clone())
      trimC_subbodies = list(reversed(trimC_subbodies))

   if trim_from_C:
      hmtcn = min(helix_trim_max, len(ub_cn) - 1)
      p = ros.core.pose.Pose()
      print(lb_cn[hmtcn] + htnie + 1)
      ros.core.pose.append_subpose_to_pose(p, pose, lb_cn[hmtcn] + htnie + 1, pose.size())
      trimN_subbodies.append(p)
      for i, (start, end) in enumerate(zip(
            reversed(lb_cn[:hmtcn]),
            reversed(ub_cn[:hmtcn]),
      )):
         p = ros.core.pose.Pose()
         stop = end if individual_pieces else pose.size()
         ros.core.pose.append_subpose_to_pose(p, pose, start + htnie + 1, stop)
         trimN_subbodies.append(p.clone())
      trimN_subbodies = list(reversed(trimN_subbodies))

   if debug:
      for i, b in enumerate(trimC_subbodies):
         print('dump body %i' % i)
         b.dump_pdb('trimC_%i.pdb' % i)

      for i, b in enumerate(trimN_subbodies):
         print('dump body %i' % i)
         b.dump_pdb('trimN_%i.pdb' % i)

   return trimN_subbodies, trimC_subbodies