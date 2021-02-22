import rpxdock as rp, numpy as np, pyrosetta.rosetta as rz
from rpxdock.rosetta.triggers_init import create_residue, Pose, AtomID, sfxn
from pyrosetta.rosetta.numeric import xyzVector_double_t as xyzVec
import pyrosetta

def make_AILV_rots():
   print('make_AILV_rots')

   from pyrosetta.rosetta import core
   rotsets = core.pack.rotamer_set.RotamerSets()
   print(type(rotsets))
   print(rotsets)

   # pose = rz.core.pose.Pose()
   pose = pyrosetta.pose_from_file(rp.data.pdbdir + '/tiny.pdb.gz')
   pose.dump_pdb('test.pdb')
   task = rz.core.pack.task.TaskFactory.create_packer_task(pose)
   print(task)

   packer_neighbor_graph = rz.core.pack.create_packer_graph(pose, sfxn, task)

   rotsets.set_task(task)

   rotsets.initialize_pose_for_rotsets_creation(pose)

   # crash!
   # rotsets.build_rotamers(pose, sfxn, packer_neighbor_graph)

   assert 0
