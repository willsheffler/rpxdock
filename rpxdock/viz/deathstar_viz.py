import sys
import numpy as np
import willutil as wu
import rpxdock as rp
from willutil import I

if 'pymol' in sys.modules:

   @wu.viz.pymol_viz.pymol_load.register(rp.DeathStar)
   def pymol_load_DeathStar(
      ds,
      state,
      name,
      pos=I,
      delprev=False,
      resrange=(0, -1),
      sym=None,
      showframes=False,
      whole=True,
      asymframes=False,
      suspend_updates=True,
      saveview=True,
      linewidth=5,
      show_aligned_ifaces=True,
      showcapcopies=False,
      showaxis=True,
      connsphere=7,
      conncyl=3,
      showcaporigin=True,
      **kw,
   ):
      # _try_to_use_pymol_objs(body, state, name, pos, hideprev, **kw)
      # return

      from pymol import cmd
      if suspend_updates: cmd.set('suspend_updates', 'on')
      if delprev: cmd.delete(f'{name}*')
      if saveview: view = cmd.get_view()

      ds.timer.checkpoint('viz')
      ds.set_dofs(ds.dofs())

      pos = ds.frames[1:]
      if asymframes or whole:
         pos = ds.frames[ds.asymunit]

      if showcaporigin:
         wu.showme(ds.frames[:1], spheres=10)
         cappos = ds.frames[0]
         c = cappos[:, 3]
         ax = cappos[:, 2] * 50
         wu.viz.show_ndarray_lines(
            np.array([c - ax, 2 * ax]).T, state=state, name=f'{name}_capaxis', spheres=0)

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

         xaln = wu.hrot([0, 0, 1], 0)
         xaln = wu.hrot([1, 0, 0], -20) @ xaln
         # xaln = wu.htrans([-35, 180, 0]) @ xaln
         xaln = wu.htrans([0, 60, 180]) @ xaln
         xaln = wu.hrot([1, 0, 0], 18) @ xaln
         xaln = wu.hrot([0, 0, 1], 100) @ xaln
         # xaln = I

         for iorig in [False]:
            ifacepos = ds.iface_positions(original=iorig)
            for inbr, (pos1, pos2) in enumerate(ifacepos):
               x1 = xaln @ wu.hinv(pos1) @ pos1
               x2 = xaln @ wu.hinv(pos1) @ pos2

               #         for iorig in [False]:
               #            xrel = ds.iface_rel_xforms(original=iorig)
               #            for inbr, x in enumerate(xrel):
               #               # xaln = I
               #               x1 = xaln @ I @ ds.fromifacecen
               #               x2 = xaln @ x @ ds.fromifacecen
               #               wu.viz.showsphere(xaln @ ds.fromifacecen @ ds.orig_iface_cen, col=[1, 0, 1], rad=4,
               # lbl=f'{name}_nbr{inbr}cen')
               for x, col, lbl in zip([x1, x2], [[1, 1, 0], [0, 1, 1]], 'AB'):
                  rp.viz.body_viz.pymol_load_Body(ds.hull, state, f'{name}_nbr{inbr}{lbl}', pos=x,
                                                  resrange=(0, -1), showframes=False, col=col,
                                                  scale=1.0, suspend_updates=False,
                                                  linewidth=linewidth, markfirst=True, **kw)
               # rp.viz.body_viz.pymol_load_Body(ds.hull, state, name + '_nbr%iB' % inbr, pos=x2,
               # resrange=(0, -1), showframes=False, col=[0, 1, 1],
               # scale=1.0, suspend_updates=False,
               # linewidth=linewidth, markfirst=True, **kw)

      rp.viz.body_viz.pymol_load_Body(ds.hull, state, name, pos=pos, resrange=(0, -1),
                                      showframes=showframes, col='rand', whitetopn=whitetopn,
                                      suspend_updates=False, linewidth=linewidth, **kw)

      if connsphere > 1e-6 or conncyl > 1e-6:
         symcom = np.concatenate([
            ds.laser.symcom(ds.frames[:1]),
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
            name=name + '_axis', spheres=0)

      if saveview: cmd.set_view(view)
      if suspend_updates: cmd.set('suspend_updates', 'off')
