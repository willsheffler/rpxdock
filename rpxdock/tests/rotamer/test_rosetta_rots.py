from rpxdock.rotamer.rosetta_rots import get_rosetta_rots

def test_rosetta_rots(dump_pdbs=False):
   get_rosetta_rots(dump_pdbs=dump_pdbs)

if __name__ == '__main__':
   test_rosetta_rots(dump_pdbs=True)