import os, glob, rpxdock as rp

def main():

   for fn in glob.glob(rp.data.bodydir + '/*'):
      os.remove(fn)

   fnames = list()
   fnames.extend(glob.glob(rp.data.pdbdir + '/*.pdb'))
   fnames.extend(glob.glob(rp.data.pdbdir + '/*.pdb.gz'))
   for fname in fnames:
      print(fname)
      n = os.path.basename(fname).rstrip('.gz').rstrip('.pdb')
      rp.dump(
         rp.Body(rp.data.pdbdir + f'/{n}.pdb'),
         rp.data.bodydir + f'/{n}.pickle',
      )

   rp.dump(
      rp.Body(rp.data.pdbdir + '/small_c3_hole.pdb', sym=3),
      rp.data.bodydir + '/small_c3_hole_sym3.pickle')

if __name__ == '__main__':
   main()