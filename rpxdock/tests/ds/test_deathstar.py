import pytest

def main():
   test_mcsample_simple()

@pytest.mark.xfail
def test_mcsample_simple():
   assert 0

if __name__ == '__main__':
   main()
