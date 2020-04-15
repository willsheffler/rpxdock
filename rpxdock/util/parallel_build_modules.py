import os, importlib, glob, time
import cppimport
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import cpu_count, current_process
import multiprocessing

def fullname_from_path(f):
   return f.replace("/util/..", "").replace("/", ".").replace(".cpp", "")

def check_needs_update(filepath):
   print('check_needs_update', filepath)
   fullname = fullname_from_path(filepath)
   module_data = cppimport.importer.setup_module_data(fullname, filepath)
   return not cppimport.checksum.is_checksum_current(module_data)

def maybe_build(f):
   try:
      fullname = fullname_from_path(f)
      print('build', fullname)
      cppimport.imp(fullname)

   except None:
      pass
   # ImportError as e:
   #  raise ValueError(f'cant import {fullname}')

def files_needing_rebuild():
   pattern = "%s/../**/*.cpp" % os.path.dirname(__file__)
   # return [f.replace('/util/..', '') for f in glob.glob(pattern) if check_needs_update(f)]
   return [f for f in glob.glob(pattern) if check_needs_update(f)]

def parallel_build_modules(cppfiles=None):
   if isinstance(current_process(), multiprocessing.context.ForkProcess):
      return

   root = os.path.dirname(__file__)

   cppfiles = files_needing_rebuild() if cppfiles is None else cppfiles
   if not cppfiles:
      return

   # print(f'{" checking ":=^80}\n', '\n'.join(cppfiles))

   fguard = os.path.join(root, ".rpxdock_parallel_build_guard")
   if os.path.exists(fguard):
      print("rpxdock.util.parallel_build_modules: build already in progress")
      nreturn

   try:
      with ProcessPoolExecutor(cpu_count()) as exe:
         fut = [exe.submit(maybe_build, g) for g in cppfiles]
         # open(fguard, "w").close()
         # print("rpxdock.util.parallel_build_modules: building")
         prev = 0
         while not time.sleep(1):
            done = [f.done() for f in fut]
            ndone = sum(done)
            if ndone != prev:
               print(f'built {ndone} of {len(fut)}')
               prev = ndone
            if all(done): break
         # for f in fut:
         # f.result()
   finally:
      print("rpxdock.util.parallel_build_modules: cleaning up")
      if os.path.exists(fguard):
         os.remove(fguard)
      print(f'{" done ":=^80}\n', '\n'.join(cppfiles))

if __name__ == "__main__":
   parallel_build_modules()
