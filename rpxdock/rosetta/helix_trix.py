from rpxdock.data import pdbdir
import rpxdock as rp
from willutil import Bunch

def append_Nhelix(userpose, alignto=7):
   from rpxdock.rosetta.triggers_init import get_pose_cached, rosetta, protocols, core
   helix = get_pose_cached('tiny.pdb.gz', pdbdir)
   first_res = rosetta.utility.vector1_unsigned_long()
   first_res.extend(range(1, alignto + 1))  #default first 7 res of pos but user can specify if more/less
   offset = helix.size() - alignto
   protocols.toolbox.pose_manipulation.superimpose_pose_on_subset_CA(helix, userpose, first_res, offset)
   core.pose.append_pose_to_pose(userpose, helix)
   return True

def append_Chelix(userpose, alignto=7):
   from rpxdock.rosetta.triggers_init import get_pose_cached, rosetta, protocols, core
   helix = get_pose_cached('tiny.pdb.gz', pdbdir)
   # Get last few res of pose (default last 7)
   last_res = rosetta.utility.vector1_unsigned_long()
   size = len(userpose.chain_sequence(1))  # Need to make this equal to just chain A/chain 1
   last_res.extend(range(size - (alignto - 1), size + 1))
   first_res = rosetta.utility.vector1_unsigned_long()
   first_res.extend(range(1, alignto + 1))  #default first 7 res of pos but user can specify if more/less
   # Make a subpose of the last few res of pose, so that the numbering can align w numbering of helix
   Cterm_subpose = core.pose.Pose()
   core.pose.pdbslice(Cterm_subpose, userpose, last_res)
   core.pose.renumber_pdbinfo_based_on_conf_chains(Cterm_subpose, fix_chains=True, start_from_existing_numbering=False)
   protocols.toolbox.pose_manipulation.superimpose_pose_on_subset_CA(helix, Cterm_subpose, first_res)
   core.pose.append_pose_to_pose(userpose, helix)
   return True

def point_in(pose, term, helixch=False):
   from rpxdock.rosetta.triggers_init import get_pose_cached, rosetta, protocols, core
   # Can pass in which chain is attached to the relevant term (integer)
   # If no helix chain given, assumes last chain in pose is helix chain
   if not helixch: helixch = pose.chain(pose.size())
   helixbeg = rosetta.protocols.geometry.center_of_mass(pose, pose.chain_begin(helixch), pose.chain_begin(helixch) + 2)
   helixend = rosetta.protocols.geometry.center_of_mass(pose, pose.chain_end(helixch) - 2, pose.chain_end(helixch))
   if term == "N":
      vterm = helixbeg - helixend
   elif term == "C":
      vterm = helixend - helixbeg
   #comv = rosetta.protocols.geometry.center_of_mass(pose, 1, pose.chain_begin(helixch)-1)
   #if vterm.z - comv.z > 0: #Points out
   if vterm.z > 0:  #Points out
      return False
   #elif vterm.z - comv.z < 0: #Points in
   elif vterm.z < 0:  #Points in
      return True

# Removes one chain from the pose. If not specified, this is the last
# chain of the pose. Otherwise, can specify which chain number to remove
def remove_helix_chain(pose, chain=None):
   from rpxdock.rosetta.triggers_init import get_pose_cached, rosetta, protocols, core
   # Get length of chain 1
   # Get start and end residues of chains to be removed
   if chain != None and pose.num_chains() <= chain:
      start_remove = pose.chain_begin(chain)
      end_remove = pose.chain_end(chain)
   elif chain == None:
      start_remove = pose.chain_begin(pose.num_chains())
      end_remove = pose.chain_end(pose.num_chains())
   else:
      return False
   pose.delete_residue_range_slow(start_remove, end_remove)
   return True

# Checks actual (N and C actual) and desired (N and C spec) direction of termini
# relative to the origin to determine if desired orientation is possible.
def limit_flip_update_pose(N_actual, C_actual, input_num=1, j=0, **kw):
   from rpxdock.rosetta.triggers_init import get_pose_cached, rosetta, protocols, core
   kw = Bunch(kw)
   N_spec = kw.termini_dir[input_num - 1][j][0]
   C_spec = kw.termini_dir[input_num - 1][j][1]
   # N_spec = kw.termini_dir[input_num -1][0]
   # C_spec = kw.termini_dir[input_num -1][1]
   if (C_spec == N_spec == None) or (N_spec == N_actual and C_spec == C_actual):
      # in correct orientation already
      if N_spec is not None or C_spec is not None:
         if kw.force_flip[input_num - 1] == True:
            error_msg=("Desired termini orientation is not possible for input "\
                        f"#{(j+1)} in list of --inputs{input_num} because the "\
                        "restrictions necessary to ensure desired orientation "\
                        "for this input are incompatible with restrictions "\
                        f"required for other input(s) in --inputs{input_num} "\
                        "to have their termini in their desired orientation. "\
                        "NOTE: Since the same sampler is used for all inputs "\
                        f"in --inputs{input_num}, it is not possible to prevent "\
                        "flipping of one input in list while forcing another "\
                        "input to flip. To work around this, submit these "\
                        "inputs to rpxdock individually, with one input for "\
                        "--inputs1, rather than submitting --inputs1 as a list.")
            return False, error_msg
         kw.flip_components[input_num -
                            1] = False  #Otherwise use original input to make body, but limit flip if not done already
      return True, None
   elif ((N_actual != None and C_actual != None)
         and ((N_actual != C_actual and N_spec == C_spec) or (N_actual == C_actual and N_spec != C_spec))):
      if len(kw.inputs[input_num - 1]) > 1:
         input_num = str(f"{(j+1)} in list of --inputs{input_num}")
      error_msg = (f"Input {input_num} cannot have its termini in the desired orientation")
      return False, error_msg
   elif ((N_spec != None and N_spec != N_actual) or (C_spec != None and C_spec != C_actual)
         or (N_spec == (not N_actual) and C_spec == (not C_actual))):
      #need to flip once, force flip
      if not kw.flip_components[input_num - 1]:
         if len(kw.inputs[input_num - 1]) > 1:
            input_num = str(f"{(j+1)} in list of --inputs{input_num}")
         # error_msg = (f"Desired termini orientation is not possible because flipping is not allowed for input {input_num}")
         error_msg = ("Desired termini orientation is not possible because "\
                     f"flipping is not allowed for input {input_num}. "\
                     "NOTE: If submitting a list of inputs, the same sampler "\
                     "is used for all inputs in that list. It is not possible "\
                     "to prevent flipping of one input in list while forcing "\
                     "another input in the list to flip. To work around this, "\
                     "submit inputs to rpxdock individually, with one input for "\
                     "--inputs1, rather than submitting --inputs1 as a list.")
         return False, error_msg
      else:  # force flip
         assert kw.flip_components[input_num - 1]  #if use force flip, must be allowed to flip
         kw.force_flip[input_num - 1] = True
         return True, None

# Wrapper-ish function to get z-direction of C terminus of pose relative to origin
def C_term_in(pose, keep_helix=False):
   from rpxdock.rosetta.triggers_init import get_pose_cached, rosetta, protocols, core
   append_Chelix(pose)
   C_point_in = point_in(pose, "C")
   if not keep_helix: remove_helix_chain(pose)
   return C_point_in

# Wrapper-ish function to get z-direction of N terminus of pose relative to origin
def N_term_in(pose, keep_helix=False):
   from rpxdock.rosetta.triggers_init import get_pose_cached, rosetta, protocols, core
   append_Nhelix(pose)
   N_point_in = point_in(pose, "N")
   if not keep_helix: remove_helix_chain(pose)
   return N_point_in

# Wrapper for helices + termini stuff
def init_termini(make_poselist=True, **kw):
   from rpxdock.rosetta.triggers_init import get_pose_cached, rosetta, protocols, core
   kw = Bunch(kw)
   poses, og_lens = [[]], [[]]
   for i in range(len(kw.inputs)):
      for j in range(len(kw.inputs[i])):
         access = kw.term_access[i][j]
         direction = kw.termini_dir[i][j]
         N_in = None
         C_in = None
         if (True in access) or (True in direction) or (False in direction):
            if len(poses) < i + 1:
               poses.append([])
            if len(og_lens) < i + 1:
               og_lens.append([])
            if type(kw.inputs[i][j]) == rosetta.core.pose.Pose:
               pose = kw.inputs[i][j]
            else:
               pose = rp.rosetta.rosetta_util.get_pose(kw.inputs[i][j])
            og_lens[i].append(pose.size())
            if direction[0] is not None:
               N_in = rp.rosetta.helix_trix.N_term_in(pose, access[0])  #N term first
            elif access[0]:
               rp.rosetta.helix_trix.append_Nhelix(pose)
            if direction[1] is not None:
               C_in = rp.rosetta.helix_trix.C_term_in(pose, access[1])  #C term
            elif access[1]:
               rp.rosetta.helix_trix.append_Chelix(pose)
            if sum(access) > 0:
               poses[i].append(pose)  #Only update pose if adding on helices to termini
            elif make_poselist:
               poses[i].append(kw.inputs[i][j])
            dir_possible, error_msg = rp.rosetta.helix_trix.limit_flip_update_pose(N_in, C_in, i + 1, j, **kw)
            if not dir_possible: raise ValueError(error_msg)
         elif make_poselist:
            # Add path to pdb to pose list, if there will be any modified poses
            # that will be used to make the body
            if len(poses) < i + 1: poses.append([])
            poses[i].append(kw.inputs[i][j])
            if len(og_lens) < i + 1: og_lens.append([])
            og_lens[i].append(None)
   return poses, og_lens
