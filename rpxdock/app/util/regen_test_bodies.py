import os, glob, rpxdock as rp

def main():

   replace = False

   if replace:
      for fn in glob.glob(rp.data.bodydir + '/*'):
         os.remove(fn)

   fnames = list()
   fnames.extend(glob.glob(rp.data.pdbdir + '/*.pdb'))
   fnames.extend(glob.glob(rp.data.pdbdir + '/*.pdb.gz'))
   for fname in fnames:
      try:
         n = os.path.basename(fname).replace('.gz', '').replace('.pdb', '')
         pdbfname = rp.data.pdbdir + f'/{n}.pdb'
         if not os.path.exists(pdbfname):
            pdbfname += '.gz'
         bodyfname = rp.data.bodydir + f'/{n}.pickle'
         if not replace and os.path.exists(bodyfname):
            print('skip', fname)
            continue
         print('make body for', fname)
         rp.dump(rp.Body(pdbfname), bodyfname)
      except Exception as e:
         print(f'regen_test_bodies fail on {fname}')
         print(e)

   if replace or not os.path.exists(rp.data.bodydir + '/small_c3_hole_sym3.pickle'):
      rp.dump(rp.Body(rp.data.pdbdir + '/small_c3_hole.pdb', sym=3), rp.data.bodydir + '/small_c3_hole_sym3.pickle')

if __name__ == '__main__':
   main()
