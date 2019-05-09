import os, sys, time, _pickle, types
from sicdock.motif import ResPairData


class Cache(dict):
    def __call__(self, fun, *args, force_reload=False, **kw):
        key = fun.__name__, repr(args), repr(kw)
        try:
            assert not force_reload
            val = self[key]
            # print("HIT", key)
        except (KeyError, AssertionError):
            print("LOADHACK CACHE MISS", key)
            val = fun(*args, **kw)
            self[key] = val
        return val

    def remove(self, fun, *args, force_reload=False, **kw):
        key = fun.__name__, repr(args), repr(kw)
        del self[key]


# print("CACHE RESET")
# os.a_very_unique_name = Cache()

if not hasattr(os, "a_very_unique_name"):
    os.a_very_unique_name = Cache()
hackcache = os.a_very_unique_name


def load_big_respairdat():
    print("LOADING BIG ResPairData")
    f = "/home/sheffler/debug/sicdock/datafiles/respairdat100.pickle"
    # f = "/home/sheffler/debug/sicdock/datafiles/respairdat_si20_rotamers.pickle"
    print("loading huge dataset")
    t = time.perf_counter()
    with open(f, "rb") as inp:
        rp = _pickle.load(inp)
    rp = ResPairData(rp)
    rp.sanity_check()
    return rp
