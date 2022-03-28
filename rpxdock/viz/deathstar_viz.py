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
      delprev=False,
      resrange=(0, -1),
      sym=None,
      showpos=False,
      whole=False,
      asymframes=False,
      suspend_updates=True,
      saveview=True,
      linewidth=3,
      **kw,
   ):
      # _try_to_use_pymol_objs(body, state, name, pos, hideprev, **kw)
      # return

      from pymol import cmd
      if suspend_updates: cmd.set('suspend_updates', 'on')
      if delprev: cmd.delete(f'{name}*')
      if saveview: view = cmd.get_view()

      ds.set_dofs(ds.dofs())

      pos = ds.frames[1:]
      if asymframes or whole:
         pos = ds.frames[ds.asymunit]
      if whole:
         pos = np.concatenate([ds.frames[:0], pos])
         pos = wu.hxform(ds.symx, pos, outerprod=True).swapaxes(0, 1).reshape(-1, 4, 4)
         pos = np.concatenate([ds.frames[:1], pos, ds.frames[-1:]])
         whitetopn = 1
         # pos = np.concatenate([pos, ds.frames[-1:]])
         # whitetopn=3

      rp.viz.body_viz.pymol_load_Body(ds.laser, state, name=f'{name}_laser', pos=ds.frames[0],
                                      suspend_updates=False, linewidth=linewidth + 1,
                                      delprev=delprev)
      if True:
         # show aligned ifaces
         ref_iface_idx = -1
         ifacepos = ds.iface_positions(ref_iface_idx=ref_iface_idx, begnbr=1)
         xaln = np.eye(4)
         xaln = wu.hinv(ifacepos[ref_iface_idx, 0]) @ ifacepos[ref_iface_idx, 1]
         xaln = wu.hrot([0, 1, 0], 90) @ wu.hinv(xaln)
         xaln = wu.hrot([1, 0, 0], 90) @ xaln
         # rp.viz.body_viz.pymol_load_Body(ds.hull, state, name + '_origin', pos=xaln, delprev=delprev,
         # resrange=(0, -1), showpos=False, col='rand',
         # whitetopn=whitetopn, suspend_updates=False,
         # linewidth=linewidth, **kw)
         for inbr, (xa, xb) in enumerate(ifacepos):
            ifpos = xaln @ wu.hinv(xa) @ xb
            rp.viz.body_viz.pymol_load_Body(ds.hull, state, name + 'nbr%i' % inbr, pos=ifpos,
                                            resrange=(0, -1), showpos=False, col='rand',
                                            whitetopn=whitetopn, scale=0.5, suspend_updates=False,
                                            linewidth=linewidth, **kw)

      rp.viz.body_viz.pymol_load_Body(ds.hull, state, name, pos=pos, resrange=(0, -1),
                                      showpos=True, col='rand', whitetopn=whitetopn,
                                      suspend_updates=False, linewidth=linewidth, **kw)

      symcom = np.concatenate([
         ds.laser.symcom(ds.frames[0]),
         ds.hull.symcom(ds.frames[1:]),
      ])
      pt1 = symcom[ds.neighbors[:, 0], ds.nbrs_internal[:, 0], :, 3]
      pt2 = symcom[ds.neighbors[:, 1], ds.nbrs_internal[:, 1], :, 3]
      vec = pt2 - pt1
      ray = np.stack([pt1, vec], axis=-1)
      wu.viz.show_ndarray_lines(ray, state, name=name + '_nbrs', scale=1.0, bothsides=True,
                                spheres=0.6, cyl=0.8)
      wu.viz.show_ndarray_lines(
         np.array([[0, 0, -150, 1], [0, 0, 300, 0]]).T, state=state, name=name + '_axis')

      if saveview: cmd.set_view(view)
      if suspend_updates: cmd.set('suspend_updates', 'off')
