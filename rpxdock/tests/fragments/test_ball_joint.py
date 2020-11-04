import rpxdock as rp, pytest
import rpxdock.fragments.ball_joint_build_db

@pytest.mark.skip
def testp_ball_joint():
   for i in range(100):
      rp.fragments.ball_joint_build_db.create_ball_joint_db()
   print('poo')
   assert 0

if __name__ == '__main__':
   test_ball_joint()
