import sys, os

# import cppimport
# somecode = cppimport.imp("somecode") #This will pause for a moment to compile the module

def main():
   assert len(sys.argv) is 2

   file = sys.argv[1]
   envinc = '/home/sheffler/miniconda3/envs/rpxdock/include/python3.7m'
   cmd = f'g++-7 -std=c++17 -w -O1 -S -Irpxdock/extern -I. -I{envinc} {file}'
   print()
   print(cmd)
   print()
   os.system(cmd)

   # cfg['include_dirs'] = ['../..', '../extern']
   # cfg['compiler_args'] = ['-std=c++17', '-w', '-Ofast']
   # cfg['dependencies'] = []
   # cfg['parallel'] = False

if __name__ == '__main__':
   main()
