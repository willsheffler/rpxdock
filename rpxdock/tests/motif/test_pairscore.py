import time, tempfile
import rpxdock as rp
from willutil import Bunch
from rpxdock.motif.pairdat import *
from rpxdock.motif.pairscore import *
from rpxdock.xbin import Xbin
import _pickle

def main():
   # make_score_files_moveme()

   # import tempfile
   # with open(f, "rb") as inp:
   # rp = ResPairData(_pickle.load(inp))
   # test_create_res_pair_score(rp, tempfile.mkdtemp())

   with tempfile.TemporaryDirectory() as tmpdir:

      rps = rp.data.small_respairscore()
      test_res_pair_score_portable_store(rps, tmpdir)

      return

      rpd = rp.data.small_respairdat()
      test_pair_score(rps)
      test_bin_get_all_data(rps)
      test_bin_score(rps)
      test_res_pair_score_pickle(rps, tmpdir)

      # rpd = rp.motif.ResPairData(rp.data.small_respairdat())

      test_create_res_pair_score(rpd, tmpdir)

def test_res_pair_score_portable_store(respairscore, tmpdir):
   rps = respairscore

   with tempfile.TemporaryDirectory() as tmpdir:
      fname = rp.motif.respairscore_to_tarball(rps, tmpdir + '/test_rps_tarball', overwrite=True)
      rps2 = rp.motif.respairscore_from_tarball(fname)
   # print(set(dir(rps)) - set(dir(rps2)))

   assert rps == rps2

def test_create_res_pair_score(respairdat, tmpdir):
   xbin = Xbin(1, 15)
   rps = create_res_pair_score(
      respairdat,
      xbin,
      min_bin_score=2,
      min_pair_score=1,
      min_ssep=10,
      use_ss_key=True,
   )
   print(rps)

def test_res_pair_score_pickle(respairscore, tmpdir):
   rps = respairscore
   with open(tmpdir + "/foo", "wb") as out:
      _pickle.dump(rps, out)
   with open(tmpdir + "/foo", "rb") as inp:
      rps2 = _pickle.load(inp)
   assert np.all(rps.rotid == rps2.rotid)
   assert np.all(rps.pdb == rps2.pdb)
   assert np.all(rps.stub == rps2.stub)
   assert rps.score_map == rps2.score_map
   assert rps.range_map == rps2.range_map

def test_pair_score(respairscore):
   rps = respairscore
   print(rps.respair.shape, rps.ssid.shape)

   keys = rps.keys
   scores = rps.score_map[keys]
   ranges = rps.range_map[keys].view("u4").reshape(-1, 2)
   nrange = ranges[:, 1] - ranges[:, 0]
   rlb = ranges[:, 0]
   rub = ranges[:, 1]

   ssi = rps.ssid[rps.respair[:, 0]]
   ssj = rps.ssid[rps.respair[:, 1]]

   for ibin in range(len(keys)):
      # print(ibin, scores[ibin], nrange[ibin], ranges[ibin])
      lb = rlb[ibin]
      ub = rub[ibin]
      assert len(np.unique(ssi[lb:ub])) == 1
      assert len(np.unique(ssj[lb:ub])) == 1

def test_bin_score(respairscore):
   keys = np.zeros(len(respairscore.keys) * 2, dtype="u8")
   keys[:len(respairscore.keys)] = respairscore.keys
   np.random.shuffle(keys)
   t = time.perf_counter()
   s = respairscore.bin_score(keys)
   t = time.perf_counter() - t
   print(f"perf {int(len(s) / t):,} {len(s):,}")

def test_bin_get_all_data(respairscore):
   # on real system: perf perrot: 434,667 perkey: 141,453

   assert not 0 in respairscore.range_map
   keys = np.zeros(len(respairscore.keys) * 2, dtype="u8")
   keys[:len(respairscore.keys)] = respairscore.keys
   np.random.shuffle(keys)

   t = time.perf_counter()
   binrots, hits = respairscore.bin_get_all_data(keys)
   t = time.perf_counter() - t
   print(len(keys), len(binrots))
   for i in range(10):
      print([x.shape for x in binrots[i]])

   assert np.all(keys[hits] > 0)
   assert np.all(keys[~hits] == 0)

   totsize = sum(len(x[0]) for x in binrots)
   print(len(binrots), totsize)
   print(f"perf perrot: {int(totsize / t):,} perkey: {int(len(binrots) / t):,}")

# def make_score_files_moveme():
#    # f = "/home/sheffler/debug/rpxdock/respairdat_si30.pickle"
#    # # f2 = "/home/sheffler/debug/rpxdock/datafiles/respairdat_si30_rotamers"
#    # f = "rpxdock/data/respairdat10_plus_xmap_rots.pickle"
#    # f2 = "rpxdock/data/respairscore10"
#    # f = "rpxdock/data/respairdat10.pickle"
#    f = "rpxdock/data/pdb_res_pair_data_si30_10.pickle"
#    f1 = "rpxdock/data/respairdat10_plus_xmap_rots.pickle"
#    f2 = "rpxdock/data/pairscore10.pickle"
#    with open(f, "rb") as inp:
#       rp = ResPairData(_pickle.load(inp))
#    rp.data["stub"] = ["resid", "hrow", ""], bb_stubs(rp.n, rp.ca, rp.c)
#    xbin = Xbin(1.0, 20)
#    add_xbin_to_respairdat(rp, xbin, min_ssep=10)
#    rotspace = get_rotamer_space()
#    add_rots_to_respairdat(rp, rotspace)
#    with open(f1, "wb") as out:
#       _pickle.dump(rp.data, out)
#    rps = create_res_pair_score(rp, min_ssep=10)
#    with open(f2, "wb") as out:
#       _pickle.dump(rps, out)
#
# def terrible_sanity_check():
#    f = "/home/sheffler/debug/rpxdock/pdb_res_pair_data_si30_pairscore.pickle"
#    with open(f, "rb") as inp:
#       rps = _pickle.load(inp)
#
#    f2 = "/home/sheffler/debug/rpxdock/pdb_res_pair_data_si30_pairscore/resdata.pickle"
#    with open(f2, "rb") as inp:
#       rd = _pickle.load(inp)
#
#    assert rps.stub.shape == rd.stub.shape
#    assert rps.xbin.cart_resl == rd.xbin.cart_resl
#    assert rps.keys.shape == rd.keys.shape
#    assert len(rps.score_map) == len(rd.keys)
#    assert len(rps.range_map) == len(rd.keys)
#    assert rps.respair.shape == rd.respair.shape
#    assert rps.aaid.shape == rd.aaid.shape
#    assert rps.ssid.shape == rd.ssid.shape
#    assert rps.rotid.shape == rd.rotid.shape
#    assert rps.stub.shape == rd.stub.shape
#    assert rps.pdb.shape == rd.pdb.shape
#    assert rps.resno.shape == rd.resno.shape
#    for i in range(len(rps.rotchi)):
#       assert list(rps.rotchi[i]) == list(rd.rotchi[i])
#    assert rps.rotlbl == rd.rotlbl
#    assert rps.id2aa.shape == rd.id2aa.shape
#    assert rps.id2ss.shape == rd.id2ss.shape
#
#    f = "/home/sheffler/debug/rpxdock/pdb_res_pair_data_si30_pairscore.pickle"
#    with open(f, "rb") as inp:
#       rps = _pickle.load(inp)
#
#    # f2 = "/home/sheffler/debug/rpxdock/pdb_res_pair_data_si30_pairscore/resdata.pickle"
#    # with open(f2, "rb") as inp:
#    # rd = _pickle.load(inp)
#
#    print(len(rps.score_map) / 1000000)
#    k, v = rps.score_map.items_array()
#
#    from time import perf_counter
#    from rpxdock.phmap import PHMap_u8f8
#
#    t = perf_counter()
#    x = rps.score_map[k]
#    t = perf_counter() - t
#    assert np.all(x == v)
#    print(t)
#    t = perf_counter()
#    p = PHMap_u8f8()
#    p[k] = v
#    t = perf_counter() - t
#    assert np.all(x == v)
#    print(t)

if __name__ == "__main__":
   main()
