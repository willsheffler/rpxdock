import copy, functools
import numpy as np
import rpxdock as rp
import willutil as wu

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
   body = body.copy_with_sym('C3')
   _test_body_viz(body)

def _test_body_viz(body, sym='icos', nfold=3):
   kw = wu.Bunch(headless=False)
   kw.headless = False
   #
   psym = 'icos'
   csym = 'c5'

   symframes = wu.sym.frames(csym)
   # pos = wu.sym.frames(psym, bbsym=csym)
   # pos = wu.sym.frames(psym, axis=[0, 0, 1], axis0=[1, 1, 1])

   sympos = wu.sym.frames(psym, axis=[0, 0, 1], asym_of=csym, bbsym=csym)
   # sympos = wu.sym.frames(psym, bbsym=csym, axis=[0, 0, 1])
   asympos = wu.homog.htrans([0, 0, 120])

   # cart = 10 * wu.sym.axes(sym, csym)
   # f[:, :, 3] += wu.homog.hdot(f, cart)

   # assert len(sympos) == 20
   print(len(sympos))
   # assert 0
   print(wu.homog.hdot(sympos[:, :, 2], [0, 0, 1]))

   # whichsub = {
   #    ('tet', 'c2'): 0,
   #    ('tet', 'c3'): 0,
   #    ('oct', 'c2'): 1,
   #    ('oct', 'c3'): 0,
   #    ('oct', 'c4'): 1,
   #    ('icos', 'c2'): 1,
   #    ('icos', 'c3'): 1,
   #    ('icos', 'c5'): 0,
   # }
   # print(len(sympos), flush=True)
   pos = sympos @ np.linalg.inv(sympos[0]) @ asympos
   # wu.viz.showme(body, name='test0', pos=pos, delprev=True, hideprev=False, linewidth=5,
   # col='rand', **kw)
   # assert 0

   # axis0 = wu.sym.axes(sym, nfold=csym)
   # pos0 = wu.homog.align_vector([0, 0, 1], axis0)
   # pos0[:, 3] += pos0[:, 2] * 100

   # pos1 = pos @ wu.homog.htrans([100, 100, 100])
   # wu.viz.showme(body, name='test0', pos=pos1, stateno=1, hideprev=True, **kw)

   # wu.viz.showme(body, name='test1', pos=pos, stateno=1, hideprev=False, col='rand', linewidth=3,
   # **kw)

   for i in range(200):
      if i % 50 == 0: print(i, flush=True)
      delta = wu.homog.rand_xform_small(len(pos), cart_sd=0.1, rot_sd=0.003)
      # delta = pos @ delta @ np.linalg.inv(pos)
      pos = pos @ delta
      wu.viz.showme(body, name='test0', pos=pos, sym=csym, delprev=True, hideprev=False,
                    stateno=i + 2, linewidth=5, col='rand', **kw)

   print('DONE 3984948', flush=True)

if __name__ == '__main__':
   main()
