import rpxdock as rp, willutil as wu, numpy as np
import pytest

def main():
   test_deathstar()
   print('DONE')

def test_deathstar():
   sym = 'oct'
   csym = 'c3'

   forig = '/home/sheffler/debug/deathstar/cage_examples/I3_AK/I3ak_orig.pdb'
   body = rp.get_body(forig)
   rad = np.linalg.norm(body.bvh_bb.com())
   xaln = wu.align_vector([1, 1, 1], [0, 0, 1])
   xaln[2, 3] = -rad
   origin = wu.hrot([0, 0, 1], wu.angle([1, 1, 1], [1, 1, 0]))
   origin[2, 3] = rad
   body = body.copy_xformed(xaln)
   body = body.copy_with_sym(csym)

   fcap = '/home/sheffler/debug/deathstar/I3ak_orig_expanded.pdb'
   cap = rp.get_body(fcap)

   # wu.viz.showme(cap)
   # assert 0

   cap = cap.copy_xformed(xaln)
   cap = cap.copy_with_sym(csym)

   ds = rp.DeathStar(body, cap, sym, csym, origin=origin)
   wu.viz.showme(ds)

if __name__ == '__main__':
   main()
