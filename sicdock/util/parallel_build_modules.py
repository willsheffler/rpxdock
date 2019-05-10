import os, importlib, glob
import cppimport
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import cpu_count, current_process
import multiprocessing


def fullname_from_path(f):
    return f.replace("/util/..", "").replace("/", ".").replace(".cpp", "")


def check_needs_update(filepath):
    fullname = fullname_from_path(filepath)
    module_data = cppimport.importer.setup_module_data(fullname, filepath)
    return not cppimport.checksum.is_checksum_current(module_data)


def maybe_build(f):
    try:
        fullname = fullname_from_path(f)
        cppimport.imp(fullname)
    except ImportError as e:
        print("error", e)


def files_needing_rebuild():
    pattern = "%s/../**/*.cpp" % os.path.dirname(__file__)
    return [f for f in glob.glob(pattern) if check_needs_update(f)]


def parallel_build_modules(cppfiles=None):
    if isinstance(current_process(), multiprocessing.context.ForkProcess):
        return

    root = os.path.dirname(__file__)

    cppfiles = files_needing_rebuild() if cppfiles is None else cppfiles
    if not cppfiles:
        return

    fguard = os.path.join(root, ".sicdock_parallel_build_guard")
    if os.path.exists(fguard):
        print("sicdock.util.parallel_build_modules: build already in progress")
        nreturn

    try:
        with ProcessPoolExecutor(cpu_count()) as exe:
            fut = [exe.submit(maybe_build, g) for g in cppfiles]
            # open(fguard, "w").close()
            print("sicdock.util.parallel_build_modules: building")
            [f.result for f in fut]
    finally:
        print("sicdock.util.parallel_build_modules: cleaning up")
        if os.path.exists(fguard):
            os.remove(fguard)


if __name__ == "__main__":
    parallel_build_modules()
