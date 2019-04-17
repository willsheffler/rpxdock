"""
usage: python runtests.py @

this script exists for easy editor integration
"""

import sys
import os
import re


def hasmain(file):
    with open(file) as inp:
        for l in inp:
            if l.startswith('if __name__ == "__main__":'):
                return True
    return False


def testfile_of(path, bname):
    print("testfile_of", path, bname)
    return re.sub("^sicdock", "sicdock/tests", path) + "/test_" + bname


def dispatch(file, pytest_args="--duration=5"):
    dispatch = {
        "rosetta.py": ["sicdock/tests/test_body.py"],
        "bvh_algo.hpp": ["sicdock/tests/bvh/test_bvh.py"],
        "bvh.cpp": ["sicdock/tests/bvh/test_bvh.py"],
    }
    file = os.path.relpath(file)
    path, bname = os.path.split(file)
    print("runtests.py dispatch", path, bname)
    if hasmain(file):
        return "PYTHONPATH=. python " + file
    if bname not in dispatch and (
        not file.endswith(".py") or not file.startswith("sicdock/")
    ):
        return "PYTHONPATH=. python " + file
    if bname in dispatch:
        if hasmain(dispatch[bname][0]):
            return "PYTHONPATH=. python " + dispatch[bname][0]
        else:
            tmp = " ".join(dispatch[bname])
            return "pytest {pytest_args} ".format(**vars()) + tmp
    if not os.path.basename(file).startswith("test_"):
        testfile = testfile_of(path, bname)
        if os.path.exists(testfile):
            if hasmain(testfile):
                return "PYTHONPATH=. python " + testfile
            return "pytest {pytest_args} {testfile}".format(**vars())
        else:
            return "PYTHONPATH=. python " + testfile
    else:
        if hasmain(file):
            return "PYTHONPATH=. python " + file
        return "pytest {pytest_args} {file}".format(**vars())
    return "pytest {pytest_args} {file}".format(**vars())


if len(sys.argv) is 1:
    cmd = "pytest"
elif len(sys.argv) is 2:
    if sys.argv[1].endswith(__file__):
        cmd = "pytest"
    else:
        cmd = dispatch(sys.argv[1])
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
print("=" * 20, "util/runtests.py done", "=" * 37)
