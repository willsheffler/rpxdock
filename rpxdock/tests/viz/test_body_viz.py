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

   axis0 = wu.sym.symaxes[sym][nfold]
   pos0 = wu.homog.align_vector([0, 0, 1], axis0)
   pos0[:, 3] += pos0[:, 2] * 100

   pos = wu.sym.frames('icos', bbsym='c3')
   pos = pos @ pos0

   wu.viz.showme(body, name='test0', pos=pos, stateno=1, hide_prev=True, **kw)
   # return

   for i in range(1000):
      if i % 10 == 0: print(i, flush=True)
      delta = wu.homog.rand_xform_small(len(pos), cart_sd=0.1, rot_sd=0.005)
      pos = delta @ pos
      wu.viz.showme(body, name='test0', pos=pos, hide_prev=True, stateno=i + 2, **kw)

   print('DONE 3984948', flush=True)

if __name__ == '__main__':
   main()
