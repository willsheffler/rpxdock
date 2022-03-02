from rpxdock.rosetta.helix_trix import *
from pyrosetta import rosetta

def test():
   return "test of util.parse_flip"

def test_processing_body(pdbfile):
   import rpxdock.rosetta.triggers_init as ros
   pose = ros.pose_from_file(pdbfile)
   print(type(pose))
   print("pre app: ", pose)
   print("")
   #print(pose.dump_pdb('/home/jenstanisl/my_rpxdock_code/rpxdock/tests/app/output/from_parse_pose.pdb'))
   #print(dir(pose))
   #print(pose)
   temp_pose = append_Chelix(pose)
   print(type(temp_pose))
   print("app pose: ", temp_pose)
   print("")
   print("og pose: ", pose)
   # print(temp_pose.dump_pdb('/home/jenstanisl/my_rpxdock_code/rpxdock/tests/app/output/Cappend_test.pdb'))
   return point_in(temp_pose, "C")

# TO DO: check if I need to keep returning the pose? -> nope
# Probably not, p sure it just gets modified directly already

def old_test_C_term_in(pdbfile, keep_helix=True):
   import rpxdock.rosetta.triggers_init as ros
   pose = ros.pose_from_file(pdbfile)
   # pose.dump_pdb('/home/jenstanisl/my_rpxdock_code/rpxdock/tests/app/output/pre_pose.pdb')
   append_Chelix(pose)
   # dumped = temp_pose.dump_pdb('/home/jenstanisl/my_rpxdock_code/rpxdock/tests/app/output/Cappend_test.pdb')
   # print("Dumped pdb C: ", dumped)
   C_point_in = point_in(pose, "C")
   print(C_point_in)
   if not keep_helix: remove_helix(pose, 1)
   # pose.dump_pdb('/home/jenstanisl/my_rpxdock_code/rpxdock/tests/app/output/pose_Cappend.pdb')
   return C_point_in

def test_C_term_in(pose, keep_helix=False):
   append_Chelix(pose)
   # dumped = temp_pose.dump_pdb('/home/jenstanisl/my_rpxdock_code/rpxdock/tests/app/output/Cappend_test.pdb')
   # print("Dumped pdb C: ", dumped)
   C_point_in = point_in(pose, "C")
   print(C_point_in)
   if not keep_helix: remove_helix_chain(pose)
   # pose.dump_pdb('/home/jenstanisl/my_rpxdock_code/rpxdock/tests/app/output/pose_Cappend.pdb')
   return C_point_in

# def old_test_N_term_in(pdbfile, keep_helix=True):
#    import rpxdock.rosetta.triggers_init as ros
#    pose = ros.pose_from_file(pdbfile)
#    temp_pose = append_Nhelix(pose)
#    # dumped = temp_pose.dump_pdb('/home/jenstanisl/my_rpxdock_code/rpxdock/tests/app/output/Nappend_test.pdb')
#    # print("Dumped pdb N: ", dumped)
#    N_point_in = point_in(pose, "N")
#    print(N_point_in)
#    if not keep_helix: remove_helix(pose, 1)
#    # temp_pose.dump_pdb('/home/jenstanisl/my_rpxdock_code/rpxdock/tests/app/output/temp_pose_Nappend.pdb')
#    # pose.dump_pdb('/home/jenstanisl/my_rpxdock_code/rpxdock/tests/app/output/pose_Nappend.pdb')
#    return N_point_in

# Gets direction of N term and also retains helix if desired
def test_N_term_in(pose, keep_helix=False):
   append_Nhelix(pose)
   # dumped = temp_pose.dump_pdb('/home/jenstanisl/my_rpxdock_code/rpxdock/tests/app/output/Nappend_test.pdb')
   # print("Dumped pdb N: ", dumped)
   N_point_in = point_in(pose, "N")
   print(N_point_in)
   if not keep_helix:
      remove_helix_chain(pose)
   # temp_pose.dump_pdb('/home/jenstanisl/my_rpxdock_code/rpxdock/tests/app/output/temp_pose_Nappend.pdb')
   # pose.dump_pdb('/home/jenstanisl/my_rpxdock_code/rpxdock/tests/app/output/pose_Nappend.pdb')
   return N_point_in
