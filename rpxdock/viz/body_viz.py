import tempfile
import numpy as np
import rpxdock as rp
import willutil as wu

@wu.viz.pymol_viz.pymol_load.register(rp.Body)
def pymol_load_Body(
      body,
      state,
      name,
      pos=np.eye(4),
      hideprev=True,
      **kw,
):
   from pymol import cmd
   cmd.disable('all')
   pos = pos.reshape(-1, 4, 4)

   ncac = body.coord
   # print('body', type(body), type(ncac), ncac.shape, ncac.strides)
   # wu.viz.pymol_cgo.showsphere([1, 2, 3, 1], 4)

   ncac = pos[:, None, None] @ ncac[None, :, :3, :, None]
   ncac = ncac.reshape(-1, 3, 4)
   # print(ncac.shape)
   wu.viz.pymol_viz.show_ndarray_line_strip(ncac, state=state, name=name, hideprev=hideprev, **kw)
   # cmd.png(f'/home/sheffler/tmp/{name}.png')
   cmd.hide('sticks')
   cmd.hide('cartoon')
   cmd.show('lines')
   cmd.set('suspend_updates', 'off')

def _try_to_use_pymol_objs(
   body,
   state,
   name,
   pos,
   hideprev,
   **kw,
):
   from pymol import cmd
   pos = pos.reshape(-1, 4, 4)
   cmd.set('suspend_updates', 'on')
   pymol_objs = cmd.get_object_list()
   name0 = name + '_0'
   # create all objs if necessary
   if not name + '_0' in pymol_objs:
      print('CREATING NEW OBJECTS', flush=True)
      cmd.delete('all')
      with tempfile.TemporaryDirectory() as tmpdir:
         # fname = tmpdir + "/" + name + ".pdb"
         fname = f'/tmp/{name}.pdb'
         body.dump_pdb(fname)
         print('loading', fname, flush=True)
         cmd.load(fname, name0)
      for i in range(len(pos) - 1):
         cmd.create(name + '_' + str(i + 1), name0)
      pymol_objs = cmd.get_object_list()
   else:
      print('CREATING NEW STATES', flush=True)
      for i in range(len(pos)):
         n = name + '_' + str(i)
         cmd.create(n, n, 1, -1)

   assert len(pymol_objs) == len(pos)
   nstates = cmd.count_states(name0)
   for i, x in enumerate(pos):
      n = name + '_' + str(i)
      assert nstates == cmd.count_states(n)
      cmd.matrix_reset(n, state=nstates)
      axis, angle = wu.homog.axis_angle_of(x)
      cmd.rotate(list(axis), np.degrees(angle), n, state=nstates, object=n, camera=0)
      cmd.translate(list(x[:3, 3]), n, state=nstates, object=n, camera=0)
      # m = np.array(cmd.get_object_matrix(n)).reshape(4, 4)
      # print(m)
      # print()

   cmd.set('suspend_updates', 'off')
   print('pymol_load_Body DONE', flush=True)
