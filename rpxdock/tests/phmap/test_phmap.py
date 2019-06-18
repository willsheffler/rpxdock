import os, _pickle
import numpy as np
from rpxdock import phmap

def ua(a):
   return np.array(a, dtype="u8")

def fa(a):
   return np.array(a, dtype="f8")

def test_phmap():
   N = 10000
   phm = phmap.PHMap_u8u8()
   k = np.random.randint(0, 2**64, N, dtype="u8")
   k = np.unique(k)
   v = np.random.randint(0, 2**64, len(k), dtype="u8")
   phm[k] = v
   assert np.all(phm[k] == v)
   shuf = np.argsort(np.random.rand(N))
   assert np.all(phm[k[shuf]] == v[shuf])

   assert phm[k[0]] == v[0]
   assert phm[k[1234]] == v[1234]
   phm[k[:10]] = 10
   assert np.all(phm[k[:10]] == 10)

   phm[77] = 77
   assert phm[77] == 77

def test_phmap_contains():
   N = 10000
   phm = phmap.PHMap_u8u8()
   k = np.random.randint(0, 2**32, N, dtype="u8")
   k = np.unique(k)
   N = len(k)
   v = np.random.randint(0, 2**32, len(k), dtype="u8")
   phm[k] = v
   # v[123] = 77
   assert np.all(phm[k] == v)
   shuf = np.argsort(np.random.rand(N))
   assert np.all(phm[k[shuf]] == v[shuf])

   assert len(phm.has(k)) == len(k)
   assert k in phm
   assert not [np.min(k) - 1, np.max(k) + 1] in phm
   assert not int(np.min(k) - 1) in phm
   assert not int(np.max(k) + 1) in phm

def test_phmap_items():
   N = 1000
   phm = phmap.PHMap_u8u8()
   k = np.random.randint(0, 2**64, N, dtype="u8")
   k = np.unique(k)
   v = np.random.randint(0, 2**64, len(k), dtype="u8")
   phm[k] = v
   assert np.all(phm[k] == v)
   shuf = np.argsort(np.random.rand(N))
   assert np.all(phm[k[shuf]] == v[shuf])

   k1 = set(k)
   k2 = set(k for k, v in phm.items())
   assert k1 == k2
   v1 = set(v)
   v2 = set(v for k, v in phm.items())
   assert v1 == v2

def test_phmap_dump_load(tmpdir):
   N = 1000
   phm = phmap.PHMap_u8u8()
   k = np.random.randint(0, 2**64, N, dtype="u8")
   k = np.unique(k)
   v = np.random.randint(0, 2**64, len(k), dtype="u8")
   phm[k] = v

   assert np.all(phm[k] == v)
   shuf = np.argsort(np.random.rand(N))
   assert np.all(phm[k[shuf]] == v[shuf])

   with open(os.path.join(tmpdir, "foo"), "wb") as out:
      _pickle.dump(phm, out)
   phm2 = phmap.PHMap_u8u8()
   with open(os.path.join(tmpdir, "foo"), "rb") as inp:
      phm2 = _pickle.load(inp)

   assert len(phm2) == len(phm)
   assert np.all(phm2[k] == v)
   assert np.all(phm2[k[shuf]] == v[shuf])

def test_phmap_cpp_roundtrip():
   N = 2
   phm = phmap.PHMap_u8u8()
   k = np.random.randint(0, 100, N, dtype="u8")
   k = np.unique(k)
   phm[k] = np.arange(len(k), dtype="u8")

   v = phm[k]
   assert 12345 not in phm

   phmap.test_mod_phmap_inplace(phm)
   assert np.all(phm[k] == v * 2)
   assert phm[12345] == phm[12345]
   assert 12345 in phm
   assert np.array(12345, "u8") in phm

def test_phmap_items_array():
   phm = phmap.PHMap_u8u8()
   phm[ua([1, 2, 3])] = ua([4, 5, 6])
   k, v = phm.items_array()
   assert set(k) == set([1, 2, 3])
   assert set(v) == set([4, 5, 6])
   assert np.all(phm[k] == v)

def test_phmap_eq():
   phm = phmap.PHMap_u8u8()
   phm[ua([1, 2, 3])] = ua([4, 5, 6])

   phm2 = phmap.PHMap_u8u8()
   phm2[ua([1, 2, 3])] = ua([4, 5, 6])
   assert phm == phm2

   phm2 = phmap.PHMap_u8u8()
   phm2[ua([1, 2, 4])] = ua([4, 5, 6])
   assert phm != phm2

   phm2 = phmap.PHMap_u8u8()
   phm2[ua([1, 2, 3])] = ua([4, 5, 0])
   assert phm != phm2

   phm2 = phmap.PHMap_u8u8()
   phm2[ua([1, 2])] = ua([4, 5])
   assert phm != phm2

   phm = phmap.PHMap_u8f8()
   phm[ua([1, 2, 3])] = fa([4, 5, 6])

   phm2 = phmap.PHMap_u8f8()
   phm2[ua([1, 2, 3])] = fa([4, 5, 6])
   assert phm == phm2

   phm2 = phmap.PHMap_u8f8()
   phm2[ua([1, 2, 4])] = fa([4, 5, 6])
   assert phm != phm2

   phm2 = phmap.PHMap_u8f8()
   phm2[ua([1, 2, 3])] = fa([4, 5, 0])
   assert phm != phm2

   phm2 = phmap.PHMap_u8f8()
   phm2[ua([1, 2])] = fa([4, 5])
   assert phm != phm2

if __name__ == "__main__":
   import tempfile

   test_phmap()
   test_phmap_items()
   test_phmap_contains()
   test_phmap_dump_load(tempfile.mkdtemp())
   test_phmap_cpp_roundtrip()
   test_phmap_items_array()
   test_phmap_eq()
