import sys
import numpy as np
import willutil as wu
import rpxdock as rp

if 'pymol' in sys.modules:

   @wu.viz.pymol_viz.pymol_load.register(rp.DeathStar)
   def pymol_load_DeathStar(
         ds,
         state,
         name,
         pos=np.eye(4),
         delprev=True,
         resrange=(0, -1),
         sym=None,
         showpos=False,
         allframes=False,
         asymframes=False,
         **kw,
   ):
      # _try_to_use_pymol_objs(body, state, name, pos, hideprev, **kw)
      # return

      from pymol import cmd
      cmd.set('suspend_updates', 'on')

      if delprev:
         cmd.delete(f'{name}*')

      symcom = np.concatenate([
         ds.cap.symcom(ds.frames[0]),
         ds.body.symcom(ds.frames[1:]),
      ])
      pt1 = symcom[ds.neighbors[:, 0], ds.nbrs_internal[:, 0], :, 3]
      pt2 = symcom[ds.neighbors[:, 1], ds.nbrs_internal[:, 1], :, 3]
      vec = pt2 - pt1
      ray = np.stack([pt1, vec], axis=-1)
      wu.viz.show_ndarray_lines(ray, state, scale=1.0, bothsides=True, spheres=5, cyl=1)

      pos = ds.allframes[1:] if allframes else ds.frames[1:]
      pos = ds.frames[ds.asymunit] if asymframes else pos

      rp.viz.body_viz.pymol_load_Body(
         ds.body,
         state,
         name,
         pos=pos,
         delprev=delprev,
         resrange=(0, -1),
         showpos=True,
         col='rand',
         **kw,
      )
      rp.viz.body_viz.pymol_load_Body(ds.cap, state, name='cap', pos=ds.frames[0])

      cmd.set('suspend_updates', 'off')
