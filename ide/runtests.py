"""
usage: python runtests.py @

this script exists for easy editor integration
"""

import sys
import os
import re
from time import perf_counter

_override = {
    "rosetta.py": ["sicdock/tests/test_body.py"],
    "bvh_algo.hpp": ["sicdock/tests/bvh/test_bvh.py"],
    "bvh.cpp": ["sicdock/tests/bvh/test_bvh.py"],
    "dockspec.py": ["sicdock/tests/search/test_gridslide.py"],
    "_orientations.hpp": ["sicdock/sampling/orientations.py"],
    "_orientations.cpp": ["sicdock/sampling/orientations.py"],
    "_orientations_test.cpp": ["sicdock/sampling/orientations.py"],
    "cookie_cutter.cpp": ["sicdock/cluster/cluster.py"],
}
_post = {
    # "tcdock.py": "pymol body1.pdb body2.pdb",
    # "test_tcdock.py": "pymol body1.pdb body2.pdb",
}


def file_has_main(file):
    with open(file) as inp:
        for l in inp:
            if l.startswith("if __name__ == "):
                return True
    return False


def testfile_of(path, bname):
    print("testfile_of", path, bname)
    t = re.sub("^sicdock", "sicdock/tests", path) + "/test_" + bname
    if os.path.exists(t):
        return t


def dispatch(file, pytest_args="--duration=5"):
    """for the love of god... clean me up"""
    file = os.path.relpath(file)
    path, bname = os.path.split(file)
    print("runtests.py dispatch", path, bname)
    if bname in _override:
        if len(_override[bname]) == 1:
            file = _override[bname][0]
            path, bname = os.path.split(file)
        else:
            assert 0

    if not bname.startswith("test_"):
        testfile = testfile_of(path, bname)
        if testfile:
            file = testfile
            path, bname = os.path.split(file)

    print(file, file_has_main(file))

    if not file_has_main(file) and bname.startswith("test_"):
        cmd = "pytest {pytest_args} {file}".format(**vars())
    elif file.endswith(".py"):
        cmd = "PYTHONPATH=. python " + file
    else:
        cmd = "pytest {pytest_args}".format(**vars())

    post = ""
    if bname in _post:
        post = _post[bname]
    return cmd, post


t = perf_counter()

post = ""
if len(sys.argv) is 1:
    cmd = "pytest"
elif len(sys.argv) is 2:
    if sys.argv[1].endswith(__file__):
        cmd = "pytest"
    else:
        cmd, post = dispatch(sys.argv[1])
else:
    print("usage: runtests.py FILE")

print("call:", sys.argv)
print("cwd:", os.getcwd())
print("cmd:", cmd)
print("=" * 20, "util/runtests.py running cmd in cwd", "=" * 23)
sys.stdout.flush()
# if 1cmd.startswith('pytest '):
os.putenv("NUMBA_OPT", "1")
# os.putenv('NUMBA_DISABLE_JIT', '1')
os.system(cmd)
print("=" * 20, "main command done", "=" * 37)
os.system(post)
t = perf_counter() - t
print("=" * 20, "util/runtests.py done, time", t, "=" * 30)
