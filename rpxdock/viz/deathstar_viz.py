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
      showframes=False,
      whole=True,
      asymframes=False,
      suspend_updates=True,
      saveview=True,
      linewidth=3,
      show_aligned_ifaces=True,
      showcapcopies=False,
      showaxis=True,
      connsphere=2,
      conncyl=1,
      showcaporigin=True,
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

      if showcaporigin:
         cappos = ds.frames[0]
         c = cappos[:, 3]
         ax = cappos[:, 2] * 100
         wu.viz.show_ndarray_lines(
            np.array([c - ax, 2 * ax]).T, state=state, name=f'{name}_capaxis')

      whitetopn = 0
      if whole:
         whitetopn = 3 if showcapcopies else 1
         capstop = 1 if showcapcopies else 0
         pos = np.concatenate([ds.frames[:capstop], pos])
         pos = wu.hxform(ds.symx, pos, outerprod=True).swapaxes(0, 1).reshape(-1, 4, 4)
         pos = np.concatenate([ds.frames[:1 - capstop], pos, ds.frames[-1:]])
         # whitetopn = 1
         # pos = np.concatenate([pos, ds.frames[-1:]])

      rp.viz.body_viz.pymol_load_Body(ds.laser, state, name=f'{name}_laser', pos=ds.frames[0],
                                      suspend_updates=False, linewidth=linewidth + 1,
                                      delprev=delprev)
      if show_aligned_ifaces:
         # show aligned ifaces
         # ifacepos = ds.iface_positions()
         # xorigin = np.eye(4)
         # xorigin = wu.hinv(ifacepos[ds.ref_iface_idx, 0]) @ ifacepos[ds.ref_iface_idx, 1]
         # xorigin = wu.hinv(xorigin)
         # xorigin = wu.hrot([0, 0, 1], 90) @ xorigin
         # xorigin = wu.htrans([-35, 180, 0]) @ xorigin

         #
         #         xaln = wu.hrot([0, 0, 1], 90)
         #         xaln = wu.htrans([-35, 180, 0]) @ xaln
         #
         #         ifpos0, xorigin0 = ds.iface_rel_xforms(original=True)
         #         print(ifpos0[3])
         #         ifpos0 = xaln @ ifpos0
         #         xorigin0 = xaln @ xorigin0
         #         rp.viz.body_viz.pymol_load_Body(ds.hull, state, name + '_nbrsorigin', pos=xorigin0,
         #                                         delprev=delprev, resrange=(0, -1), showframes=False,
         #                                         col=[1, 0, 0], whitetopn=0, suspend_updates=False,
         #                                         linewidth=linewidth, **kw)
         #         for inbr, x in enumerate(ifpos0):
         #            rp.viz.body_viz.pymol_load_Body(
         #               ds.hull, state, name + 'nbr%i' % inbr, pos=x, resrange=(0, -1), showframes=False,
         #               col=[1, 0,
         #                    0], whitetopn=0, scale=1.0, suspend_updates=False, linewidth=linewidth, **kw)
         #
         #         #
         #
         #         ifpos, xorigin = ds.iface_rel_xforms()
         #         print(ifpos[3])
         #         ifpos = xaln @ ifpos
         #         xorigin = xaln @ xorigin
         #
         #         # foo = wu.hinv(xorigin) @ xorigin0
         #         # xorigin = xorigin @ foo
         #         # ifpos = ifpos @ foo
         #
         #         rp.viz.body_viz.pymol_load_Body(ds.hull, state, name + '_nbrsorigin', pos=xorigin,
         #                                         delprev=delprev, resrange=(0, -1), showframes=False,
         #                                         col='rand', whitetopn=1, suspend_updates=False,
         #                                         linewidth=linewidth, **kw)
         #         for inbr, x in enumerate(ifpos):
         #            rp.viz.body_viz.pymol_load_Body(ds.hull, state, name + 'nbr%i' % inbr, pos=x,
         #                                            resrange=(0, -1), showframes=False, col='rand',
         #                                            whitetopn=1, scale=1.0, suspend_updates=False,
         #                                            linewidth=linewidth, **kw)

         xaln = wu.hrot([0, 0, 1], 0)
         xaln = wu.hrot([1, 0, 0], -50) @ xaln
         # xaln = wu.htrans([-35, 180, 0]) @ xaln
         xaln = wu.htrans([0, 60, 180]) @ xaln
         xaln = wu.hrot([1, 0, 0], -0) @ xaln

         for iorig in (True, False):
            ifacepos = ds.iface_positions(original=iorig)
            for inbr, (pos1, pos2) in enumerate(ifacepos):
               x1 = xaln @ np.eye(4)  #wu.hinv(pos1) @ pos1
               x2 = xaln @ wu.hinv(pos1) @ pos2
               wot = 0 if iorig else 1
               rp.viz.body_viz.pymol_load_Body(ds.hull, state, name + 'nbr%i' % inbr, pos=x1,
                                               resrange=(0, -1), showframes=False, col=[1, 0, 0],
                                               whitetopn=wot, scale=1.0, suspend_updates=False,
                                               linewidth=linewidth, **kw)
               rp.viz.body_viz.pymol_load_Body(ds.hull, state, name + 'nbr%i' % inbr, pos=x2,
                                               resrange=(0, -1), showframes=False, col=[1, 0, 0],
                                               whitetopn=wot, scale=1.0, suspend_updates=False,
                                               linewidth=linewidth, **kw)

      #

      rp.viz.body_viz.pymol_load_Body(ds.hull, state, name, pos=pos, resrange=(0, -1),
                                      showframes=showframes, col='rand', whitetopn=whitetopn,
                                      suspend_updates=False, linewidth=linewidth, **kw)

      if connsphere > 1e-6 or conncyl > 1e-6:
         symcom = np.concatenate([
            ds.laser.symcom(ds.frames[0]),
            ds.hull.symcom(ds.frames[1:]),
         ])
         pt1 = symcom[ds.neighbors[:, 0], ds.nbrs_internal[:, 0], :, 3]
         pt2 = symcom[ds.neighbors[:, 1], ds.nbrs_internal[:, 1], :, 3]
         vec = pt2 - pt1
         ray = np.stack([pt1, vec], axis=-1)
         wu.viz.show_ndarray_lines(ray, state, name=name + '_nbrs', scale=1.0, bothsides=True,
                                   spheres=connsphere, cyl=conncyl)
      if showaxis:
         wu.viz.show_ndarray_lines(
            np.array([[0, 0, -150, 1], [0, 0, 300, 0]]).T, col=[(1, 0, 0)], state=state,
            name=name + '_axis')

      if saveview: cmd.set_view(view)
      if suspend_updates: cmd.set('suspend_updates', 'off')
