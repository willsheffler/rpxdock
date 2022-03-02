from rpxdock.rosetta.rosetta_util import *
from rpxdock.rosetta.helix_trix import *
from rpxdock.data import pdbdir
from rpxdock.rosetta.triggers_init import core, protocols, rosetta
import pyrosetta

import argparse

#appends helix to both N and C termini
def test_append_helix(top7, helix):
   #print(top7)
   #top7.dump_pdb('before.pdb')
   #print(helix)
   #helix.dump_pdb('helix.pdb')
   #top7.dump_pdb("top7.pdb")
   #print(top7.size())
   #print(top7.size())
   #print(helix.size())

   # create new pose of last few residues (7 for now)
   last_res = range(top7.size()-6, top7.size()+1)
   #print(last_res)
   subset_res = rosetta.utility.vector1_unsigned_long()
   subset_res.extend(last_res)
   Cterm_subpose = core.pose.Pose()
   core.pose.pdbslice(Cterm_subpose, top7, subset_res)
   print(core.pose.renumber_pdbinfo_based_on_conf_chains(Cterm_subpose, fix_chains = True, start_from_existing_numbering = False))
   Cterm_subpose.dump_pdb("subpose.pdb")

   # Select residues to align to
   first_res = rosetta.utility.vector1_unsigned_long()
   first_res.extend(range(1,8))
   #print(first_res)

   #core.scoring.superimpose_pose(helix, top7, idmap, 0.0001, True)
   #core.scoring.calpha_superimpose_pose(helix, top7[1,22])   
   #aligns helix to Cterm:
   protocols.toolbox.pose_manipulation.superimpose_pose_on_subset_CA(helix, Cterm_subpose, first_res)
   #helix.dump_pdb('C_aligned.pdb')
   core.pose.append_pose_to_pose(top7, helix)
#   top7.dump_pdb('Cappend.pdb')

   #to N term, offset by 21-7 bc helix is 21 residues long and we only want to align last 7 to ref
   protocols.toolbox.pose_manipulation.superimpose_pose_on_subset_CA(helix, top7, first_res, 21-7)
   #helix.dump_pdb('N_aligned.pdb')
   #print(core.pose.append_pose_to_pose(top7, helix))
   core.pose.append_pose_to_pose(top7, helix)
   #top7.dump_pdb('Nappend.pdb')

   return top7

def test_append_Nhelix(pose, fullpath = None):
   # new_pose = append_Nhelix(pose)
   append_Nhelix(pose)
   if fullpath == None:
      # new_pose.dump_pdb("Nappend_test.pdb")
      pose.dump_pdb("Nappend_test.pdb")
   else:
      path_lst = fullpath.split("/")
      path_lst[-1] = "N_append_" + path_lst[-1]
      save_path = "/".join(path_lst)
      save_name = "/home/jenstanisl/test_rpx/test_appendhelix/dump_test_pdbs/" + path_lst[-1]
      # new_pose.dump_pdb(save_name)
      pose.dump_pdb(save_name)
      return save_name

def test_append_Chelix(pose, fullpath = None):
   # new_pose = append_Chelix(pose)
   append_Chelix(pose)
   if fullpath == None:
      # new_pose.dump_pdb("Cappend_test.pdb")
      pose.dump_pdb("Cappend_test.pdb")
   else:
      path_lst = fullpath.split("/")
      path_lst[-1] = "C_append_" + path_lst[-1]
      save_path = "/".join(path_lst)
      save_name = "/home/jenstanisl/test_rpx/test_appendhelix/dump_test_pdbs/" + path_lst[-1]
      # new_pose.dump_pdb(save_name)
      pose.dump_pdb(save_name)
      return save_name

def test_point_in(pose, term):
   is_in = point_in(pose,term)
   # print(term+" in :", is_in)
   return is_in

def test_remove(pose, path = None):
   #new_pose = remove_helix(pose, 1)
   new_pose = remove_helix(pose, 1)
   if path == None:
      new_pose.dump_pdb("both_nohelix.pdb")
   else:
      path_lst = path.split("/")
      path_lst[-1] = "no_helix_" + path_lst[-1]
      save_path = "/".join(path_lst)
      save_name = "/home/jenstanisl/test_rpx/test_appendhelix/dump_test_pdbs/" + path_lst[-1]
      new_pose.dump_pdb(save_name)
      return save_name

def parse_args():
   parser = argparse.ArgumentParser(description="check term direction")
   parser.add_argument("--pdb", type=str, required=True, help="pdb of interest")
   parser.add_argument("--term", type=str, required=True, help="terminus to add helix to")
   args = parser.parse_args()
   return args.pdb, args.term

if __name__ == '__main__':
   from rpxdock.rosetta.triggers_init import get_pose_cached
   pdb_path, term = parse_args()
   pose = pyrosetta.io.pose_from_pdb(pdb_path)
   if term == "N":
      save_name = test_append_Nhelix(pose, pdb_path)
      N_pose = pyrosetta.io.pose_from_pdb(save_name)
      # point_in = test_point_in(N_pose, "N")
      point_in = test_point_in(pose, "N")
   elif term == "C":
      save_name = test_append_Chelix(pose, pdb_path)
      C_pose = pyrosetta.io.pose_from_pdb(save_name)
      # point_in = test_point_in(C_pose, "C")
      point_in = test_point_in(pose, "C")
   elif term == "both":
      test_append_Nhelix(pose, pdb_path)
      N_point_in = point_in = test_point_in(pose, "N")
      test_append_Chelix(pose, pdb_path)
      C_point_in = test_point_in(pose, "C")
   else:
      print("Can't do that"); return False
   remove_path = test_remove(pose, pdb_path)
   f = open("/home/jenstanisl/test_rpx/test_appendhelix/dump_test_pdbs/output.txt", "a")
   f.write(" ".join([pdb_path, term, str(point_in), str(save_name), str(remove_path)]))
   f.close()
   return True



'''if __name__ == '__main__':
   from rpxdock.rosetta.triggers_init import get_pose_cached
   pdb_path, term = parse_args()

   dhr = get_pose_cached('dhr64.pdb.gz', pdbdir)
   top7 = get_pose_cached('top7.pdb.gz', pdbdir)
   #helix = get_pose_cached('tiny.pdb.gz', pdbdir)
   #test_append_helix(top7, helix)
   #test_append_helix(dhr, helix)
#   test_append_Nhelix(dhr)
#   N_dhr = pyrosetta.io.pose_from_pdb("Nappend_test.pdb")
#   test_point_in(N_dhr, "N")
#   
#   dhr = get_pose_cached('dhr64.pdb.gz', pdbdir)
#   test_append_Chelix(dhr)
#   C_dhr = pyrosetta.io.pose_from_pdb("Cappend_test.pdb")
#   test_point_in(C_dhr, "C")

   #pose = pyrosetta.io.pose_from_pdb("Cappend_test.pdb")
   #pose = pyrosetta.io.pose_from_pdb("Nappend_test.pdb")
   pose = pyrosetta.io.pose_from_pdb("C2_ank1C2G3_asu_appended.pdb")
   test_remove(pose)'''

