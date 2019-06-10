try:
   import matplotlib.pyplot as plt
   import matplotlib as mpl
   can_plot = True
except ImportError:
   can_plot = False

if can_plot:
   figsize0 = (16, 9)
   font = {"family": "normal", "weight": "bold", "size": 22}
   mpl.rc("font", **font)
   fig = None
   axes = None
   pos = 0
   rmjr = False

   def subplots(x, y, figsize=None, rowmajor=False, **kw):
      figsize = figsize if figsize else figsize0
      global fig, axes, pos, rmjr
      pos = 0
      rmjr = rowmajor
      figsize = (figsize0[0] * x, figsize0[1] * y)
      fig, axes = plt.subplots(x, y, figsize=figsize, sharex="none", sharey="none", **kw)

   def get_plotter(*args, **kw):
      global fig, axes, pos, rmjr
      if fig:
         if rmjr:
            n = len(axes)
            plotter = axes[pos % n, pos // n]
         else:
            n = len(axes[0])
            plotter = axes[pos // n, pos % n]
         pos += 1
         return True, fig, plotter
      else:
         if "figsize" not in kw:
            kw["figsize"] = figsize0
         return False, plt.figure(**kw), plt

   def scatter(*args, title="", show=True, xscale="linear", lines=False, figsize=figsize0, **kw):
      subplot, fig, plotter = get_plotter(*args, figsize=figsize, **kw)
      plotter.scatter(*args, **kw)
      if lines:
         plt.plot(*args, **kw)

      plotter.xscale(xscale)
      if subplot:
         plotter.set_title(title, fontsize=24)
      if not subplot and show:
         plotter.show()

   def hist(*args, title="", show=True, **kw):
      subplot, fig, plotter = get_plotter(*args, **kw)

      plotter.hist(*args, **kw)
      plotter.set_title(title, fontsize=24)

      if not subplot and show:
         plotter.show()

   def show():
      plt.show()
      global fig, axes, pos
      fig, axes, pos = None, None, 0
