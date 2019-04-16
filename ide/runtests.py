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


def dispatch(file, pytest_args="--duration=5"):
    dispatch = {"rosetta.py": ["body.py"]}
    file = os.path.relpath(file)
    path, bname = os.path.split(file)
    if hasmain(file):
        return "PYTHONPATH=. python " + file
    if not file.endswith(".py") or not file.startswith("tcdock/"):
        return "PYTHONPATH=. python " + file
    if not os.path.basename(file).startswith("test_"):
        if bname in dispatch:
            if hasmain(path + "/" + dispatch[bname][0]):
                return "PYTHONPATH=. python " + path + "/" + dispatch[bname][0]
            else:
                return "pytest {pytest_args} ".format(**vars()) + " ".join(
                    (os.path.join(path, n) for n in dispatch[bname])
                )
        else:
            testfile = path + "/test_" + bname
            if os.path.exists(testfile):
                if hasmain(testfile):
                    return "PYTHONPATH=. python " + testfile
                return "pytest {pytest_args} {testfile}".format(**vars())
            else:
                return "PYTHONPATH=. python " + testfile
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
