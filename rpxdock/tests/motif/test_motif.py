from rpxdock.motif import *
from rpxdock.xbin import Xbin, xbin_util as xu
import pytest

def pair_key_ss_py(xbin, resi, resj, ss, stub):
   ssi = ss[resi]
   ssj = ss[resj]
   kij = xu.key_of_selected_pairs(xbin, resi, resj, stub, stub)
   kji = xu.key_of_selected_pairs(xbin, resj, resi, stub, stub)
   assert kij.dtype == np.uint64
   ssi = ssi.astype("u8")
   ssj = ssj.astype("u8")
   kijss = np.bitwise_or(np.left_shift(ssi, 62), np.left_shift(ssj, 60))
   kjiss = np.bitwise_or(np.left_shift(ssj, 62), np.left_shift(ssi, 60))
   kssij = np.bitwise_or(kij, kijss)
   kssji = np.bitwise_or(kji, kjiss)
   # assert np.all(np.right_shift(kssij, 62) == ssi)
   # assert np.all(np.right_shift(kssji, 62) == ssj)
   # assert np.all(np.right_shift(np.left_shift(kssij, 2), 62) == ssj)
   # assert np.all(np.right_shift(np.left_shift(kssji, 2), 62) == ssi)
   # assert np.all(np.right_shift(np.left_shift(kssij, 4), 4) == kij)
   # assert np.all(np.right_shift(np.left_shift(kssji, 4), 4) == kji)
   return kssij, kssji

def get_pair_keys_ss_py(rp, xbin, min_ssep):
   mask = (rp.p_resj - rp.p_resi).data >= min_ssep
   resi = rp.p_resi.data[mask]
   resj = rp.p_resj.data[mask]
   stub = rp.stub.data
   ss = rp.ssid.data
   kij = np.zeros(len(rp.p_resi), dtype="u8")
   kji = np.zeros(len(rp.p_resi), dtype="u8")
   kij[mask], kji[mask] = pair_key_ss_py(xbin, resi, resj, ss, stub)
   return kij, kji

def test_jagged_bin():
   kij = np.random.randint(1, 9e2, int(9e3)).astype("u8")
   kji = np.random.randint(1, 9e2, int(9e3)).astype("u8")
   k = np.concatenate([kij, kji])
   order, binkey, binrange = cpp.jagged_bin(k)
   kord = k[order]
   assert np.all(np.diff(kord) >= 0)
   lb, ub = binrange.view("u4").reshape(-1, 2).swapaxes(0, 1)
   assert np.sum(ub == lb) == 0, "not empty"
   assert np.all(kord[lb] == binkey)
   assert np.all(kord[ub - 1] == binkey)

def test_jagged_bin_zero():
   N = 1000
   kij = np.random.randint(1, 9e2, N).astype("u8")
   kji = np.random.randint(1, 9e2, N).astype("u8")
   kij[np.random.randint(0, 2, N) == 0] = 0
   kji[np.random.randint(0, 2, N) == 0] = 0
   k = np.concatenate([kij, kji])
   order, binkey, binrange = cpp.jagged_bin(k)
   kord = k[order]
   assert np.all(np.diff(kord) >= 0)
   lb, ub = binrange.view("u4").reshape(-1, 2).swapaxes(0, 1)
   assert np.sum(ub == lb) == 0, "not empty"
   assert np.all(kord[lb] == binkey)
   assert np.all(kord[ub - 1] == binkey)

def test_pair_key_ss(respairdat):
   xbin = Xbin(1, 20)
   rp = respairdat
   kij, kji = get_pair_keys(rp, xbin, min_pair_score=0, min_ssep=0, use_ss_key=True)
   kij2, kji2 = get_pair_keys_ss_py(rp, xbin, min_ssep=0)
   assert np.all(kij == kij2)
   assert np.all(kji == kji2)
   ss = rp.ssid.data
   resi = rp.p_resi.data
   resj = rp.p_resj.data
   assert np.all(np.right_shift(kij, 62) == ss[resi])
   assert np.all(np.right_shift(kji, 62) == ss[resj])
   assert np.all(np.right_shift(np.left_shift(kij, 2), 62) == ss[resj])
   assert np.all(np.right_shift(np.left_shift(kji, 2), 62) == ss[resi])

def test_pair_key(respairdat):
   xbin = Xbin(1, 20)
   rp = respairdat
   kij, kji = get_pair_keys(rp, xbin, min_pair_score=0, min_ssep=0, use_ss_key=False)
   kij2 = xu.key_of_selected_pairs(xbin, rp.p_resi.data, rp.p_resj.data, rp.stub.data,
                                   rp.stub.data)
   kji2 = xu.key_of_selected_pairs(xbin, rp.p_resj.data, rp.p_resi.data, rp.stub.data,
                                   rp.stub.data)
   assert np.all(kij == kij2)
   assert np.all(kji == kji2)

# @pytest.mark.slow
def test_respairdat_addrots(respairdat):
   rotspace = get_rotamer_space()
   add_rots_to_respairdat(respairdat, rotspace)
   m, s = check_rotamer_deviation(respairdat, rotspace, quiet=1)
   assert m < 20
   assert s < 20
   xbin = Xbin(cart_resl=1, ori_resl=20)
   add_xbin_to_respairdat(respairdat, xbin, min_pair_score=0, min_ssep=10, use_ss_key=True)

def test_remove_redundant_pdbs():
   pdbs = "12AS 155C 16PK 16VP 1914 19HC 1A04 1A05 1A0C 1A0D".split()
   assert 3 == len(remove_redundant_pdbs(pdbs, 30))
   assert 4 == len(remove_redundant_pdbs(pdbs, 40))
   assert 5 == len(remove_redundant_pdbs(pdbs, 50))
   assert 7 == len(remove_redundant_pdbs(pdbs, 70))
   assert 9 == len(remove_redundant_pdbs(pdbs, 90))
   assert 10 == len(remove_redundant_pdbs(pdbs, 95))

# if __name__ == "__main__":
#     # respairdat =
#     # test_jagged_bin()
#     test_jagged_bin_zero()
#     # test_respairdat()
