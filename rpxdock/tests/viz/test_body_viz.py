import functools
import pytest
import numpy as np
import rpxdock as rp
import willutil as wu
from willutil.homog import htrans, hrot

@functools.singledispatch
def sdtest(body):
   assert 0

@sdtest.register(rp.Body)
def _(body):
   print('BODY!', body.coord.shape)

def _test_body_sd():
   sdtest(body)
   body2 = body.copy_with_sym('C3')
   sdtest(body2)

def main():
   body = rp.data.get_body('C3_1na0-1_1')
   body = body.copy_with_sym('c3')
   origin = htrans([0, 0, 113]) @ hrot([1, 0, 0], 180) @ hrot([0, 0, 1], 95)
   _test_body_viz(body, 'icos', 'c3', origin=origin)
   return
   # wu.showme(body, psym='icos', csym='c3', pos=origin)

   forig = '/home/sheffler/debug/deathstar/cage_examples/I3_AK/I3ak_orig.pdb'
   body = rp.get_body(forig)
   rad = np.linalg.norm(body.bvh_bb.com())
   xaln = wu.homog.align_vector([1, 1, 1], [0, 0, 1])
   xaln[2, 3] = -rad
   body = body.copy_xformed(xaln)
   body = body.copy_with_sym('c3')
   origin = hrot([0, 0, 1], wu.homog.angle([1, 1, 1], [1, 1, 0]))
   origin[2, 3] = rad
   _test_body_viz(body, 'icos', 'c3', origin=origin)

def _test_body_viz(body, psym='icos', csym='c3', origin=np.eye(4)):
   pytest.importorskip('pymol')
   kw = wu.Bunch(headless=True)
   kw.headless = False
   #
   # psym = 'icos'
   # csym = 'c3'
   vsym = None

   flb, fub, fnum = 1, -1, 2
   if psym == 'tet' and csym == 'c3':
      fub, fnum = None, 1

   symframes = wu.sym.frames(csym)
   # pos = wu.sym.frames(psym, bbsym=csym)
   # pos = wu.sym.frames(psym, axis=[0, 0, 1], axis0=[1, 1, 1])

   sympos1 = wu.sym.frames(psym, axis=[0, 0, 1], asym_of=csym, bbsym=csym)
   sympos2 = wu.sym.frames(psym, axis=[0, 0, 1], asym_of=csym, bbsym=csym, asym_index=1)
   # sympos1 = sympos1[flb:fub]
   # sympos2 = sympos2[flb:fub]
   # sympos1 = wu.sym.frames(psym, bbsym=csym, axis=[0, 0, 1])

   # wu.showme(body)
   # wu.showme(body, pos=origin)
   # wu.showme(body, pos=wu.sym.frames('icos', axis=[0, 0, 1]) @ origin)
   # assert 0s

   pos1 = sympos1 @ np.linalg.inv(sympos1[0]) @ origin
   pos2 = sympos2 @ np.linalg.inv(sympos2[0]) @ origin
   pos = np.concatenate([pos1[flb:fub], pos2])

   symcom = body.symcom(pos1)
   wu.showme(symcom)
   comdist1 = body.symcomdist(pos1[flb:fub], mask=True)
   comdist2 = body.symcomdist(pos1[flb:fub], pos2)
   if len(pos1[flb:fub]) > 1:
      # print(pos1.shape, comdist1)
      a1, _, a2, _ = np.where(comdist1 < np.min(comdist1) + 0.001)
   else:
      a1, a2 = np.array([], dtype='i'), np.array([], dtype='i')
   b1, _, b2, _ = np.where(comdist2 < np.min(comdist2) + 0.001)
   nbrs = np.stack([np.concatenate([a1, b1]), np.concatenate([a2, b2 + len(pos1) - fnum])]).T

   # print(pos[0])
   # return

   wu.showme(body, name='test0', pos=pos, delprev=False, hideprev=False, linewidth=5, col='rand',
             nbrs=nbrs, **kw)

   sympos = wu.sym.frames(psym, axis=[0, 0, 1], bbsym=csym, asym_of=csym)
   pos = sympos @ np.linalg.inv(sympos[0]) @ origin
   wu.showme(body, name='test0', pos=pos, delprev=False, hideprev=False, linewidth=5, col='rand',
             **kw)
   sympos = wu.sym.frames(psym, axis=[0, 0, 1], bbsym=csym)
   pos = sympos @ np.linalg.inv(sympos[0]) @ origin
   wu.showme(body, name='test0', pos=pos, delprev=False, hideprev=False, linewidth=5, col='rand',
             **kw)
   assert 0

   # pseudo-symmetrize
   # print(wu.sym.frames(csym).shape, pos[flb:fub].shape)
   sympos = wu.sym.frames(csym).reshape(-1, 1, 4, 4) @ pos[flb:fub].reshape(1, -1, 4, 4)
   sympos = np.concatenate([pos[:1], sympos.reshape(-1, 4, 4), pos[-1:]])
   sympos = pos

   # showvecfrompoint(Vec(0, 0, 400), Vec(0, 0, -200))
   for i in range(1):
      if i % 50 == 0: print(i, flush=True)
      delta = wu.homog.rand_xform_small(len(sympos), cart_sd=0.1, rot_sd=0.003)
      # delta = sympos @ delta @ np.linalg.inv(sympos)
      sympos = sympos @ delta
      wu.showme(body, name='test0', pos=sympos, sym=vsym, delprev=True, hideprev=False,
                stateno=i + 2, linewidth=5, col='rand', **kw)

   # print('DONE 3984948', flush=True)

if __name__ == '__main__':
   main()
