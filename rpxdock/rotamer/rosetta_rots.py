import numpy as np, rpxdock as rp, xarray as xr
import rpxdock.rosetta.triggers_init
from pyrosetta import Pose, get_score_function
from pyrosetta.rosetta import core, numeric, utility, ObjexxFCL
from pyrosetta.rosetta.core.pack.task import TaskFactory
from rpxdock.rosetta.triggers_init import rts_fastd

_rosetta_known_atypes = [4, 5, 6, 32]
_rosetta_atype_to_rpx_atype = -np.ones(1000, dtype='i4')
_rosetta_atype_to_rpx_atype[4] = 0
_rosetta_atype_to_rpx_atype[5] = 1
_rosetta_atype_to_rpx_atype[6] = 2
_rosetta_atype_to_rpx_atype[32] = 3

def rosetta_atype_to_rpx_atype(at):
   return _rosetta_atype_to_rpx_atype[at]

def get_designable_positions_sasa(pose):
   # use sasa
   pass

def get_rosetta_rots(pose, ires, designable_positions='all'):
   if designable_positions == 'auto':
      designable_positions = get_designable_positions_sasa(pose)
   if designable_positions == 'all':
      designable_positions = np.ones(pose.size(), dtype=np.bool_)

def ala_to_virtCB(pose):
   for ires, res in enumerate(pose.residues):
      if res.name() == 'ALA':
         core.pose.replace_pose_residue_copying_existing_coordinates(
            pose, ires + 1, rts_fastd.name_map('ALAvirtCB'))
   # print(pose.sequence())
   # print(pose.residue(1))
   # print(pose.residue(1).atom_type(5))
   # assert 0

def get_rosetta_rots(
   pose,
   whichres='all',
   sfxn=None,
   dump_pdbs=False,
   include_unknown_atom_types=False,
   allowed_aas=[
      core.chemical.AA.aa_ala,
      core.chemical.AA.aa_ile,
      core.chemical.AA.aa_leu,
      core.chemical.AA.aa_val,
   ],
   debug=False,
   extra_rots=[False, False, False, False],
   extrachi_nnb_cutoff=0,
   dump_rotamers=False,
   disable_restypes=['ALAvirtCB'],
):
   pose = pose.clone()  # will be modified, don't screw the caller!
   if sfxn is None: sfxn = rp.rosetta.get_score_function()

   if whichres == all: whichres = np.arange(1, pose.size() + 1)
   residues_allowed_to_be_packed = utility.vector1_bool(pose.size())
   for i in range(1, pose.size() + 1):
      residues_allowed_to_be_packed[i] = False
      if i in whichres:
         residues_allowed_to_be_packed[i] = True

   aas = utility.vector1_bool(20)
   for iaa in range(1, 21):
      aas[iaa] = (iaa in allowed_aas)

   task = TaskFactory.create_packer_task(pose)
   task.restrict_to_residues(residues_allowed_to_be_packed)
   # task.initialize_extra_rotamer_flags_from_command_line()
   sfxn.setup_for_packing(pose, task.repacking_residues(), task.designing_residues())
   badrestypes = utility.vector1_string(len(disable_restypes))
   for i, t in enumerate(disable_restypes):
      badrestypes[i + 1] = t
   for ires in range(1, len(pose) + 1):
      task.nonconst_residue_task(ires).restrict_absent_canonical_aas(aas)
      task.nonconst_residue_task(ires).disable_restypes(badrestypes)

      task.nonconst_residue_task(ires).or_ex1(extra_rots[0])
      task.nonconst_residue_task(ires).or_ex2(extra_rots[1])
      task.nonconst_residue_task(ires).or_ex3(extra_rots[2])
      task.nonconst_residue_task(ires).or_ex4(extra_rots[3])
      task.nonconst_residue_task(ires).and_extrachi_cutoff(extrachi_nnb_cutoff)

   packer_neighbor_graph = core.pack.create_packer_graph(pose, sfxn, task)

   rotsets = core.pack.rotamer_set.RotamerSets()
   rotsets.set_task(task)
   rotsets.build_rotamers(pose, sfxn, packer_neighbor_graph)
   rotsets.prepare_sets_for_packing(pose, sfxn)

   for ires in whichres:
      rotset = rotsets.rotamer_set_for_residue(ires)

      energies1b = utility.vector1_float(rotset.num_rotamers())
      rotset.compute_one_body_energies(pose, sfxn, task, packer_neighbor_graph, energies1b)
      energies1b = np.array(energies1b, dtype='f4')

      if debug: print("total rots: ", rotset.num_rotamers())

      rotdata = rp.Bunch(coords=list(), resnum=list(), rotnum=list(), atomnum=list(),
                         resname=list(), atomname=list(), atomtype=list(),
                         rosetta_atom_type_index=list(), onebody=energies1b)

      # for (Size irot = 1 irot <= rotset.num_rotamers() irot++) {
      for irot in range(1, rotset.num_rotamers() + 1):
         rot = rotset.rotamer(irot)

         if dump_rotamers:
            rotpose = Pose()
            rotpose.append_residue_by_jump(rot, 1)
            fn = f"nrot{rotset.num_rotamers():04}_{rot.name()}_{irot:02}.pdb"
            print(fn)
            rotpose.dump_pdb(fn)

         for ia in range(1, rot.natoms() + 1):
            rat = rot.atom_type_index(ia)
            aname = rot.atom_name(ia)
            if aname.strip() in 'N H CA HA C O'.split():
               continue
            if rat not in _rosetta_known_atypes and not include_unknown_atom_types:
               continue

            at = rosetta_atype_to_rpx_atype(rat)
            xyz = rot.xyz(ia)
            rotdata.coords.append([xyz.x, xyz.y, xyz.z, 1])
            rotdata.resnum.append(ires)
            rotdata.rotnum.append(irot)
            rotdata.atomnum.append(ia)
            rotdata.atomtype.append(at)
            rotdata.resname.append(rot.name())
            rotdata.atomname.append(rot.atom_name(ia))
            rotdata.rosetta_atom_type_index.append(rat)
            # print(irot, rot.name(), ia, rot.atom_name(ia), at, xyz)

   rotdata.coords = np.array(rotdata.coords, dtype='f4')
   rotdata.rotnum = np.array(rotdata.rotnum, dtype='i4')
   rotdata.atomnum = np.array(rotdata.atomnum, dtype='i4')
   rotdata.atomtype = np.array(rotdata.atomtype, dtype='i4')
   rotdata.resname = np.array(rotdata.resname, dtype='<U3')
   rotdata.atomname = np.array(rotdata.atomname, dtype='<U4')
   rotdata.rosetta_atom_type_index = np.array(rotdata.rosetta_atom_type_index, dtype='i4')

   return rotdata
