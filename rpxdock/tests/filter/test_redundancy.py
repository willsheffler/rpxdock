from concurrent.futures import ThreadPoolExecutor
import numpy as np
import rpxdock as rp
import willutil as wu

def test_redundancy(hscore, body_cageA, body_cageB):
   kw = rp.app.defaults()
   kw.wts = wu.Bunch(ncontact=0.3, rpx=1.0)
   kw.beam_size = 2e4
   kw.max_bb_redundancy = 1.0
   kw.max_delta_h = 9999
   kw.nout_debug = 0
   kw.nout_top = 0
   kw.nout_each = 0
   kw.score_only_ss = 'H'
   kw.max_trim = 0
   kw.executor = ThreadPoolExecutor(min(4, kw.ncpu / 2))

   spec = rp.search.DockSpec2CompCage('T33')
   sampler = rp.sampling.hier_multi_axis_sampler(spec, [50, 60])
   result = rp.search.make_multicomp([body_cageA, body_cageB], spec, hscore, rp.hier_search, sampler, **kw)

   xforms = result.xforms.copy()
   scores = result.scores.copy()

   xforms3 = xforms.copy()
   symframes = wu.sym.frames(spec.sym)

   iuniq = rp.filter_redundancy(xforms, result.bodies[0], scores, symframes=spec.sym, **kw)
   assert len(iuniq) == len(result)

   iuniq4 = rp.filter_redundancy(xforms, result.bodies[0], scores, symframes=spec.sym, **kw.sub(max_bb_redundancy=4.0))
   iuniq3 = rp.filter_redundancy(xforms, result.bodies[0], scores, symframes=spec.sym, **kw.sub(max_bb_redundancy=3.0))
   iuniq2 = rp.filter_redundancy(xforms, result.bodies[0], scores, symframes=spec.sym, **kw.sub(max_bb_redundancy=2.0))
   iuniq1 = rp.filter_redundancy(xforms, result.bodies[0], scores, symframes=spec.sym, **kw.sub(max_bb_redundancy=1.0))
   assert len(iuniq4) <= len(iuniq3) <= len(iuniq2) <= len(iuniq1)

   xforms2 = np.concatenate([xforms, xforms], axis=0)
   scores2 = np.concatenate([scores, scores], axis=0)
   iuniq = rp.filter_redundancy(xforms2, result.bodies[0], scores2, symframes=spec.sym, **kw)
   assert len(iuniq) == len(result)

   randsymframes = wu.sym.frames(spec.sym)[np.random.choice(12, len(result))]
   xforms2randsym = xforms2.copy()
   xforms1b = wu.hxform(randsymframes, result.xforms[:, 1])
   assert 0, 'RPXDOCK test_redundancy'
   xforms2randsym[:len(xforms1b), 1] = xforms1b
   assert not np.allclose(xforms2randsym, xforms2)
   iuniq = rp.filter_redundancy(xforms2randsym, result.bodies[0], scores2, symframes=spec.sym, **kw)
   assert len(iuniq) == len(result)

def main():
   hscore = rp.data.small_hscore()
   body1 = rp.data.get_body('T33_dn2_asymA')
   body2 = rp.data.get_body('T33_dn2_asymB')
   test_redundancy(hscore, body1, body2)

if __name__ == '__main__':
   main()