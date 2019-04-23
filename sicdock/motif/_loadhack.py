import os, sys, time, _pickle
from sicdock.motif import ResPairData


def load_which_takes_forever():
    f = "/home/sheffler/debug/sicdock/datafiles/respairdat100.pickle"
    # f = "/home/sheffler/debug/sicdock/datafiles/respairdat_si20_rotamers.pickle"
    print("loading huge dataset")
    t = time.perf_counter()
    with open(f, "rb") as inp:
        rp = _pickle.load(inp)
    rp = ResPairData(rp)
    rp.sanity_check()
    return rp


if hasattr(os, "a_very_unique_name"):
    respairdat = os.a_very_unique_name
else:
    respairdat = load_which_takes_forever()
    os.a_very_unique_name = respairdat
