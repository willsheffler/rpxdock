import os, _pickle
import numpy as np
import sicdock as sic
from sicdock.rotamer import get_rotamer_space
from sicdock.util import Bunch
from sicdock.sampling import xform_hier_guess_sampling_covrads
from sicdock.xbin import smear
from sicdock.util import load, dump
from sicdock.motif import Xmap, ResPairScore


class HierScore:
    def __init__(self, files, maxdis=8):
        if isinstance(files[0], str):
            if "_noSS_" in files[0]:
                assert all("_noSS_" in f for f in files)
                self.use_ss = False
            else:
                assert all("_SS_" in f for f in files)
                self.use_ss = True
            assert "base" in files[0]
            self.base = load(files[0])
            self.hier = [load(f) for f in files[1:]]
            self.resl = list(h.attr.cart_extent for h in self.hier)
        else:
            self.base = files[0]
            self.hier = list(files[1:])
            self.use_ss = self.base.attr.opts.use_ss_key
            assert all(self.use_ss == h.attr.cli_args.use_ss_key for h in self.hier)
        self.maxdis = [maxdis + h.attr.cart_extent for h in self.hier]

    def score(self, iresl, body1, body2, contact_weight=0.1):
        pairs = body1.contact_pairs(body2, self.maxdis[iresl])
        xbin = self.hier[iresl].xbin
        phmap = self.hier[iresl].phmap
        if self.use_ss:
            pair_score = xbin.ssmap_of_selected_pairs(
                phmap, pairs, body1.ssid, body2.ssid, body1.stub, body2.stub
            )
        else:
            pair_score = xbin.map_of_selected_pairs(
                phmap, pairs, body1.stub, body2.stub
            )
        # pair_score = pair_score[pair_score > 0]
        # print(len(pair_score), np.quantile(pair_score, [0, 0.5, 0.9, 1]))
        return np.sum(pair_score) + contact_weight * len(pair_score)

    def iresls(self):
        return [i for i in range(len(self.hier))]

    def score_all(self, x):
        return np.stack([h[x] for h in self.hier])

    def score_by_resl(self, resl, x_or_k):
        if resl < 0 or resl > self.resl[0] * 2:
            raise ValueError("resl out of bounds")
        iresl = np.argmin(np.abs(resl - self.resl))
        return self.hier[iresl][x_or_k]

    def score_base(self, x_or_k):
        return self.base[x_or_k]


def create_xbin_even_nside(cart_resl, ori_resl, max_cart):
    xbin = sic.Xbin(cart_resl, ori_resl, max_cart)
    if xbin.ori_nside % 2 != 0:
        xbin = sic.xbin.create_Xbin_nside(cart_resl, xbin.ori_nside + 1, max_cart)
    return xbin


def make_and_dump_hier_score_tables(rp, **_):
    o = Bunch(_)
    fnames = list()

    resls, xhresl, nbase_nm3 = xform_hier_guess_sampling_covrads(**o)
    # xbin_base = sic.Xbin(o.base_cart_resl, ORI_RESL, o.xbin_max_cart)
    xbin_base = create_xbin_even_nside(
        o.base_cart_resl, o.base_ori_resl, o.xbin_max_cart
    )
    rps = sic.motif.create_res_pair_score(rp, xbin_base, **o)
    rps.attr.opts = o
    rps.attr.nbase_nm3 = nbase_nm3
    rps.attr.xhresl = xhresl

    print(o.base_cart_resl, resls[-1][0])
    sstag = "SS" if o.use_ss_key else "noSS"
    ftup = sstag, _rmzero(f"{o.min_pair_score}"), _rmzero(f"{o.min_bin_score}")
    fnames.append(o.out_prefix + "_%s_p%s_b%s_base.pickle" % ftup)
    dump(rps, fnames[-1])

    if len(o.smear_params) == 1:
        o.smear_params = o.smear_params * len(resls)
    assert len(o.smear_params) == len(resls)
    for ihier, (cart_extent, ori_extent) in enumerate(resls):
        if o.only_do_hier >= 0 and o.only_do_hier != ihier:
            continue
        smearrad, exhalf = o.smear_params[ihier]
        cart_resl = cart_extent / (smearrad * 3 - 1 + exhalf)
        ori_nside = xbin_base.ori_nside
        if ori_extent / xbin_base.ori_resl > 1.8:
            ori_nside //= 2
        if smearrad == 0 and exhalf == 0:
            cart_resl = cart_extent

        xbin = sic.xbin.create_Xbin_nside(cart_resl, ori_nside, o.xbin_max_cart)
        basemap, *_ = sic.motif.create_res_pair_score_map(rp, xbin, **o)
        assert basemap.xbin == xbin

        if smearrad > 0:
            if o.smear_kernel == "flat":
                kern = []
            if o.smear_kernel == "x3":  # 1/R**3 uniform in R
                grid_r2 = xbin.grid6.neighbor_sphere_radius_square_cut(smearrad, exhalf)
                kern = 1 - (np.arange(grid_r2 + 1) / grid_r2) ** 1.5
            smearmap = smear(
                xbin,
                basemap.phmap,
                radius=smearrad,
                extrahalf=exhalf,
                oddlast3=1,
                sphere=1,
                kernel=kern,
            )
        else:
            smearmap = basemap.phmap
        sm = sic.motif.Xmap(xbin, smearmap, rehash_bincens=True)
        ori_lever_extent = ori_extent * np.pi / 180 * o.sampling_lever
        sm.attr.hresl = np.sqrt(cart_extent ** 2 + ori_lever_extent ** 2)
        sm.attr.cli_args = o
        sm.attr.smearrad = smearrad
        sm.attr.exhalf = exhalf
        sm.attr.cart_extent = cart_extent
        sm.attr.ori_extent = ori_extent
        sm.attr.use_ss_key = o.use_ss_key

        print(
            f"{ihier} {smearrad} {exhalf}",
            f"cart {cart_extent:6.2f} {cart_resl:6.2f}",
            f"ori {ori_extent:6.2f} {xbin.ori_resl:6.2f}",
            f"nsmr {len(smearmap)/1e6:5.1f}M",
            f"base {len(basemap)/1e3:5.1f}K",
            f"xpnd {len(smearmap) / len(basemap):7.1f}",
        )
        fname = o.out_prefix + "_%s_p%s_b%s_hier%i_%s_%i_%i.pickle" % (
            *(sstag, _rmzero(f"{o.min_pair_score}"), _rmzero(f"{o.min_bin_score}")),
            *(ihier, "K" + o.smear_kernel, smearrad, exhalf),
        )
        fnames.append(fname)
        dump(sm, fname)
    return fnames


def _rmzero(a):
    if a[-1] == "0" and "." in a:
        b = a.rstrip("0")
        return b.rstrip(".")
    return a


def hscore_test_data(tag, path=None):
    # pref = os.path.dirname(__file__) + "/../data/hscore"
    pref = "/home/sheffler/debug/sicdock/hier/"
    return HierScore(
        [
            pref + "pdb_res_pair_data_si30_10_rots_noSS_p0.5_b1_base.pickle",
            pref + "pdb_res_pair_data_si30_10_rots_noSS_p0.5_b1_hier0_Kflat_1_0.pickle",
            pref + "pdb_res_pair_data_si30_10_rots_noSS_p0.5_b1_hier1_Kflat_1_0.pickle",
            pref + "pdb_res_pair_data_si30_10_rots_noSS_p0.5_b1_hier2_Kflat_1_0.pickle",
            pref + "pdb_res_pair_data_si30_10_rots_noSS_p0.5_b1_hier3_Kflat_1_0.pickle",
            pref + "pdb_res_pair_data_si30_10_rots_noSS_p0.5_b1_hier4_Kflat_1_0.pickle",
        ]
    )


def hscore_test_data_ss(tag, path=None):
    # pref = os.path.dirname(__file__) + "/../data/hscore"
    pref = "/home/sheffler/debug/sicdock/hier/"
    return HierScore(
        [
            pref + "pdb_res_pair_data_si30_10_rots_SS_p0.5_b1_base.pickle",
            pref + "pdb_res_pair_data_si30_10_rots_SS_p0.5_b1_hier0_Kflat_1_0.pickle",
            pref + "pdb_res_pair_data_si30_10_rots_SS_p0.5_b1_hier1_Kflat_1_0.pickle",
            pref + "pdb_res_pair_data_si30_10_rots_SS_p0.5_b1_hier2_Kflat_1_0.pickle",
            pref + "pdb_res_pair_data_si30_10_rots_SS_p0.5_b1_hier3_Kflat_1_0.pickle",
            pref + "pdb_res_pair_data_si30_10_rots_SS_p0.5_b1_hier4_Kflat_1_0.pickle",
        ]
    )
