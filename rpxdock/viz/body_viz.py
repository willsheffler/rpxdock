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
      delprev=True,
      resrange=(0, -1),
      sym=None,
      **kw,
):
   # _try_to_use_pymol_objs(body, state, name, pos, hideprev, **kw)
   # return

   from pymol import cmd
   cmd.set('suspend_updates', 'on')
   # cmd.disable('all')

   if delprev:
      cmd.delete(f'{name}*')

   sym = wu.sym.frames(sym)

   pos = pos.reshape(-1, 4, 4)
   bbsnfold = len(body) // len(body.asym_body)
   breaks = bbsnfold * len(pos) * len(sym)
   breaks_groups = bbsnfold

   coord = body.coord  # [resrange[0]:resrange[1]]
   coord = pos[:, None, None] @ coord[None, :, 1, :, None]
   coord = coord.reshape(-1, 4)

   coord = sym[:, None] @ coord[None, :, :, None]
   coord = coord.reshape(-1, 4)

   wu.viz.show_ndarray_line_strip(coord, state=state, name=name, breaks=breaks,
                                  breaks_groups=breaks_groups, **kw)

   # cmd.hide('sticks')
   # cmd.hide('cartoon')
   # cmd.show('lines')
   cmd.set('suspend_updates', 'off')

def _try_to_use_pymol_objs(
   body,
   state,
   name,
   pos,
   hideprev,
   **kw,
):
   '''this is slow and incorrct... not sure why'''
   from pymol import cmd
   pos = pos.reshape(-1, 4, 4)
   cmd.set('suspend_updates', 'on')
   pymol_objs = cmd.get_object_list()
   name0 = name + '_0'
   # create all objs if necessary
   if not name0 in pymol_objs:
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
