import os, _pickle, threading
from itertools import repeat
import numpy as np
import sicdock as sic
from sicdock.rotamer import get_rotamer_space
from sicdock.util import Bunch
from sicdock.sampling import xform_hier_guess_sampling_covrads
from sicdock.xbin import smear
from sicdock.xbin import xbin_util as xu
from sicdock.util import load, dump
from sicdock.motif import Xmap, ResPairScore, marginal_max_score
from sicdock.bvh import bvh_collect_pairs_range_vec, bvh_collect_pairs_vec


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
        self.tl = threading.local()

    def score(self, iresl, body1, body2, wcontact=0):
        return self.scorepos(iresl, body1, body2, body1.pos, body2.pos, wcontact)

    def scorepos(self, iresl, body1, body2, pos1, pos2, wts, bounds=None):
        pos1, pos2 = pos1.reshape(-1, 4, 4), pos2.reshape(-1, 4, 4)
        if not bounds:
            bounds = ([-2e9], [2e9], [-2e9], [2e9])
        if len(bounds) is 2:
            bounds += ([-2e9], [2e9])

        pairs, lbub = bvh_collect_pairs_range_vec(
            body1.bvh_cen, body2.bvh_cen, pos1, pos2, self.maxdis[iresl], *bounds
        )
        # pairs, lbub = bvh_collect_pairs_vec(
        #     body1.bvh_cen, body2.bvh_cen, pos1, pos2, self.maxdis[iresl]
        # )

        xbin = self.hier[iresl].xbin
        phmap = self.hier[iresl].phmap
        ssstub = body1.ssid, body2.ssid, body1.stub, body2.stub
        ssstub = ssstub if self.use_ss else ssstub[2:]
        fun = xu.ssmap_pairs_multipos if self.use_ss else xu.map_pairs_multipos

        pscore = fun(xbin, phmap, pairs, *ssstub, lbub, pos1, pos2)

        lbub1, lbub2, idx1, idx2, res1, res2 = marginal_max_score(lbub, pairs, pscore)

        scores = np.zeros(max(len(pos1), len(pos2)))
        for i, (lb, ub) in enumerate(lbub):
            side1 = np.sum(res1[lbub1[i, 0] : lbub1[i, 1]])
            side2 = np.sum(res2[lbub2[i, 0] : lbub2[i, 1]])
            mscore = side1 + side2
            # mscore = np.sum(pscore[lb:ub])
            # mscore = np.log(np.sum(np.exp(pscore[lb:ub])))
            scores[i] = wts.rpx * mscore + wts.ncontact * (ub - lb)

        return scores

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
