from sicdock.xbin import xmap


def test_hash_cpp():
    xmap.test_phmap()
    xmap.test_phmap2(1000, 10)


if __name__ == "__main__":
    test_hash_cpp()
