import rpxdock.rosetta.triggers_init
from pyrosetta import Pose, get_score_function
from pyrosetta.rosetta import core, numeric, utility, ObjexxFCL
from pyrosetta.rosetta.core.pack.task import TaskFactory

def get_rosetta_rots(dump_pdbs=False):
   chem_manager = core.chemical.ChemicalManager
   rts = chem_manager.get_instance().residue_type_set('fa_standard')

   pose = Pose()
   core.pose.make_pose_from_sequence(pose, 'AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA', 'fa_standard',
                                     False, False)
   nres = 31
   cenres = 16

   for i in range(1, pose.size() + 1):
      pose.set_phi(i, -47)
      pose.set_psi(i, -57)
      pose.set_omega(i, 180)

   residues_allowed_to_be_packed = utility.vector1_bool(nres)
   for i in range(1, nres):
      residues_allowed_to_be_packed[i] = False
   residues_allowed_to_be_packed[cenres] = True

   aas = utility.vector1_bool(20)
   for i in range(1, 21):
      aas[i] = False
   aas[core.chemical.AA.aa_leu] = True
   aas[core.chemical.AA.aa_ile] = True
   aas[core.chemical.AA.aa_val] = True

   pose.dump_pdb("refhelix.pdb")

   sfxn = get_score_function()

   task = TaskFactory.create_packer_task(pose)
   task.restrict_to_residues(residues_allowed_to_be_packed)
   task.initialize_extra_rotamer_flags_from_command_line()
   sfxn.setup_for_packing(pose, task.repacking_residues(), task.designing_residues())
   task.nonconst_residue_task(cenres).restrict_absent_canonical_aas(aas)

   task.nonconst_residue_task(cenres).or_ex1(True)
   task.nonconst_residue_task(cenres).or_ex2(True)
   task.nonconst_residue_task(cenres).and_extrachi_cutoff(0)

   packer_neighbor_graph = core.pack.create_packer_graph(pose, sfxn, task)

   rotsets = core.pack.rotamer_set.RotamerSets()
   rotsets.set_task(task)
   rotsets.build_rotamers(pose, sfxn, packer_neighbor_graph)
   rotsets.prepare_sets_for_packing(pose, sfxn)

   rotset = rotsets.rotamer_set_for_residue(cenres)
   print("total rots: ", rotset.num_rotamers())

   # for (Size irot = 1 irot <= rotset.num_rotamers() irot++) {
   for irot in range(1, rotset.num_rotamers() + 1):
      rot = rotset.rotamer(irot)
      rotpose = Pose()
      rotpose.append_residue_by_jump(rot, 1)
      fn = f"{rot.name()}_{irot:02}.pdb"
      print(fn)
      rotpose.dump_pdb(fn)
