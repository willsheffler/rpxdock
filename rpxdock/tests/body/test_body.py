from decimal import MAX_PREC
from os import supports_bytes_environ
from rpxdock.motif.frames import stub_from_points
import _pickle
from time import perf_counter
import numpy as np, rpxdock as rp, rpxdock.homog as hm
from rpxdock.body import Body, get_trimming_subbodies

from pyrosetta import rosetta as ros, pose_from_file
import rpxdock.rosetta.triggers_init

# ic.configureOutput(includeContext=True)

def main():
   test_body_copy_xform(rp.data.get_body('tiny'))
   test_body_ss_info()
   # test_body_create()

def test_body_ss_info():
   kw = rp.app.defaults()
   kw.helix_trim_max = 6
   pdb = rp.data.datadir + '/pdb/C3_1na0-1_1.pdb.gz'
   pose = pose_from_file(pdb)
   rp.rosetta.triggers_init.assign_secstruct(pose)
   body = Body(pose, **kw)
   # get_trimming_subbodies(body, pose)

def test_body_create():
   pdb = rp.data.datadir + '/pdb/C3_1na0-1_1.pdb.gz'
   b0 = Body(pdb)
   b1 = Body(pdb, allowed_res=None)
   b2 = Body(pdb, allowed_res=lambda x: {1, 2, 3})

   ic("len(b0.cen)", len(b0.cen))
   ic("len(b1.cen)", len(b1.cen))
   ic("len(b2.cen)", len(b2.cen))

   assert len(b0.cen) == 108  #21 old number for some reason doesn't match
   assert len(b1.cen) == 108  #21
   assert len(b2.cen) == 3

   assert len(b0.nterms) == 1
   assert len(b0.cterms) == 1

def test_body(C2_3hm4, C3_1nza, sym1=2, sym2=3):
   body1 = Body(C2_3hm4, sym1)
   body2 = Body(C3_1nza, sym2)
   print(body2.nres)
   assert body1.bvh_bb.max_id() == body1.nres - 1
   assert body1.bvh_cen.max_id() == body1.nres - 1
   assert body2.bvh_bb.max_id() == body2.nres - 1
   assert body2.bvh_cen.max_id() == body2.nres - 2  # GLY

   resl = 5
   samp1 = range(0, 360 // sym1, resl)
   samp2 = range(0, 360 // sym2, resl)
   samp3 = range(-10, 11, resl)
   samp3 = [[np.cos(d / 180 * np.pi), 0, np.sin(d / 180 * np.pi)] for d in samp3]

   r1 = hm.hrot([0, 0, 1], 1, degrees=True)
   best, bestpos = -9e9, None
   t = perf_counter()
   totslide = 0
   nsamp, nhit = 0, 0
   for a1 in samp1:
      for a2 in samp2:
         for dirn in samp3:
            body1.move_to_center()
            body2.move_to_center()
            tmp = perf_counter()
            d = body1.slide_to(body2, dirn)
            totslide += perf_counter() - tmp
            nsamp += 1
            if d < 9e8:
               nhit += 1
               p = body1.contact_pairs(body2, maxdis=8.0)
               if len(p) > 0:
                  p2 = body1.positioned_cen()[p[:, 0]]
                  p3 = body2.positioned_cen()[p[:, 1]]
                  assert np.max(np.linalg.norm(p3 - p2, axis=1)) < 8
               if len(p) > best:
                  best = len(p)
                  bestpos = body1.pos.copy(), body2.pos.copy()
         body2.move_by(r1)
      body1.move_by(r1)
   t = perf_counter() - t
   print("best", best, "time", t, "rate", nsamp / t, "hitfrac", nhit / nsamp)
   print(
      "totslide",
      totslide,
      "slide/s",
      nsamp / totslide,
      "sqrt(npair)",
      np.sqrt(len(body1.ss) * len(body2.ss)),
      len(body1.ss),
      len(body2.ss),
   )
   # print(bestpos[0])
   # print(bestpos[1])
   body1.move_to(bestpos[0])
   body2.move_to(bestpos[1])
   # body1.dump_pdb_from_bodies("body1.pdb")
   # body2.dump_pdb_from_bodies("body2.pdb")

def test_body_pickle(C3_1nza, tmpdir):
   b = Body(C3_1nza)
   with open(tmpdir + "/a", "wb") as out:
      _pickle.dump(b, out)
   with open(tmpdir + "/a", "rb") as inp:
      b2 = _pickle.load(inp)

   assert np.allclose(b.coord, b2.coord)
   assert np.allclose(b.pos, b2.pos)
   assert np.allclose(b.cen, b2.cen)
   assert b.sym == b2.sym
   assert b.nfold == b2.nfold
   assert np.all(b.seq == b2.seq)
   assert np.all(b.ss == b2.ss)
   assert np.allclose(b.chain, b2.chain)
   assert np.allclose(b.resno, b2.resno)
   assert np.allclose(b.bvh_bb.centers(), b2.bvh_bb.centers())
   assert np.allclose(b.bvh_cen.centers(), b2.bvh_cen.centers())

def test_body_copy_sym(body_tiny):
   c2 = body_tiny.copy_with_sym('C2')
   rot = hm.hrot([0, 0, 1], np.pi)
   rotated = rot @ body_tiny.coord[:, :, :, None]
   # assert np.allclose(rotated.squeeze(), c2.coord[14:28])
   assert np.allclose(rotated.squeeze(), c2.coord[21:])

def test_body_copy_xform(body_tiny):
   x = hm.hrot([1, 1, 1], np.pi / 3) @ hm.htrans([1, 0, 0])
   b2 = body_tiny.copy_xformed(x)
   rotated = x @ body_tiny.coord[:, :, :, None]
   assert np.allclose(rotated.squeeze(), b2.coord)

#function to test modifications to body by adding helices aligned with termini
def test_body_with_terminal_helices(C2_3hm4, C3_1nza, helix):
   from pyrosetta.rosetta.core import pose
   inp1 = pose.Pose().assign(C2_3hm4)
   inp2 = pose.Pose().assign(C2_3hm4)
   inp3 = pose.Pose().assign(C3_1nza)
   inp4 = pose.Pose().assign(C3_1nza)

   kw = rp.app.defaults()
   # Clean inputs list with poses that will not be modified
   clean_inputs = [[C2_3hm4], [C2_3hm4], [C3_1nza, C3_1nza]]
   kw.inputs = [[inp1], [inp2], [inp3, inp4]]
   # Test variety of conditions for access
   kw.term_access = [[[False, False]], [[False, True]], [[True, False], [True, True]]]
   kw.termini_dir = [[[None, None]], [[True, None]], [[False, None], [None, True]]]
   kw.flip_components = [True] * len(kw.inputs)
   kw.force_flip = [False] * len(kw.inputs)
   poses, og_lens = rp.rosetta.helix_trix.init_termini(**kw)

   if len(poses) > 0:
      assert len(poses) == len(og_lens) == len(kw.inputs)
      bodies = [[
         rp.Body(pose2, og_seqlen=og2, modified_term=modterm2, **kw)
         for pose2, og2, modterm2 in zip(pose1, og1, modterm)
      ]
                for pose1, og1, modterm in zip(poses, og_lens, kw.term_access)]

   # Make bodies from original inputs - no terminal modifications
   og_bodies = [[rp.Body(i2, **kw) for i2 in i1] for i1 in clean_inputs]

   # Compare 3 bodies: with appended helices (body), with no modifications (og),
   # and after helix removal (new)
   for i, b in enumerate(bodies):
      for j in range(len(b)):
         # body, og, new = b[0], og_bodies[i][0], b[0].copy_exclude_term_res()
         body, og, new = b[j], og_bodies[i][j], b[j].copy_exclude_term_res()
         assert sum(body.allowed_residues) == body.og_seqlen
         assert int(body.nres) == body.og_seqlen + (helix.size() * sum(body.modified_term))
         assert kw.term_access[i][j] == body.modified_term
         assert len(new.seq) == body.og_seqlen == og.nres
         assert np.array_equal(new.seq, og.seq)
         for k in range(0, len(og.orig_coords)):
            assert np.array_equal(new.orig_coords[k], og.orig_coords[k])

if __name__ == "__main__":
   from rpxdock.rosetta.triggers_init import get_pose_cached
   main()

   # from tempfile import mkdtemp

   #f1 = "C2_3hm4_1.pdb.gz"
   #f2 = "C3_1nza_1.pdb.gz"
   # f1 = "/home/sheffler/scaffolds/big/C2_3jpz_1.pdb"

   # f2 = "/home/sheffler/scaffolds/big/C3_3ziy_1.pdb"
   # f1 = "/home/sheffler/scaffolds/wheel/C3.pdb"
   # f2 = "/home/sheffler/scaffolds/wheel/C5.pdb"

   #pose1 = get_pose_cached(f1)
   #pose2 = get_pose_cached(f2)
   # test_body(pose1, pose2)

   # test_body_pickle(f2, mkdtemp())

   # b = rp.data.get_body('tiny')
   # test_body_copy_sym(b)
   # test_body_copy_xform(b)

   # # nres  306  309 sqnpair  307 new 17743/s orig 13511/s
   # # nres  728 1371 sqnpair  999 new  8246/s orig  4287/s
   # # nres 6675 8380 sqnpair 7479 new  8629/s orig   627/s

   #test_body_create()

   # test_body_ss_info()

   # test_body_with_terminal_helices(pose1, pose2, helix=get_pose_cached('tiny.pdb.gz')
   #test_body_with_terminal_helices(f1, f2, helix=get_pose_cached('rpxdock/data/pdb/tiny.pdb.gz'))
