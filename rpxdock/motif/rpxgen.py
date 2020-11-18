import logging, numpy as np, rpxdock as rp
from concurrent.futures import ThreadPoolExecutor
from rpxdock.bvh import bvh_collect_pairs_range_vec, bvh_collect_pairs_vec

log = logging.getLogger(__name__)

def create_xbin_even_nside(cart_resl, ori_resl, max_cart):
   xbin = rp.Xbin(cart_resl, ori_resl, max_cart)
   if xbin.ori_nside % 2 != 0:
      xbin = rp.xbin.create_Xbin_nside(cart_resl, xbin.ori_nside + 1, max_cart)
   return xbin

def make_and_dump_hier_score_tables(pairdat, **kw):
   kw = rp.Bunch(kw)
   fnames = list()

   resls, xhresl, nbase_nm3 = rp.sampling.xform_hier_guess_sampling_covrads(**kw)
   # xbin_base = rp.Xbin(kw.base_cart_resl, ORI_RESL, kw.xbin_max_cart)
   xbin_base = create_xbin_even_nside(kw.base_cart_resl, kw.base_ori_resl, kw.xbin_max_cart)
   pairscore = rp.motif.create_res_pair_score(pairdat, xbin_base, **kw)
   pairscore.attr.opts = kw
   pairscore.attr.nbase_nm3 = nbase_nm3
   pairscore.attr.xhresl = xhresl

   log.debug(f'base_cart_resl {kw.base_cart_resl} resls[-1][0] {resls[-1][0]}')
   sstag = "SSdep" if kw.use_ss_key else "SSindep"
   ftup = kw.score_only_ss, kw.score_only_aa, sstag, _rmzero(f"{kw.min_pair_score}"), _rmzero(
      f"{kw.min_bin_score}")
   fnames.append(kw.output_prefix + "%s_%s_%s_p%s_b%s_base.pickle" % ftup)
   rp.dump(pairscore, fnames[-1])

   if len(kw.smear_params) == 1:
      kw.smear_params = kw.smear_params * len(resls)
   assert len(kw.smear_params) == len(resls)

   exe = rp.util.InProcessExecutor()
   if kw.nthread > 1:
      exe = ThreadPoolExecutor(kw.nthread)
   if kw.nthread < 0:
      exe = ThreadPoolExecutor(kw.ncpu)
   futures = list()
   for ihier, (cart_extent, ori_extent) in enumerate(resls):
      if kw.only_do_hier >= 0 and kw.only_do_hier != ihier:
         continue
      f = futures.append(
         exe.submit(make_hscore_single, pairdat, ihier, xbin_base, cart_extent, ori_extent, sstag,
                    **kw))
   fnames.extend(f.result() for f in futures)
   return [f.replace('//', '/') for f in fnames]

def make_hscore_single(pairdat, ihier, xbin_base, cart_extent, ori_extent, sstag, **kw):
   kw = rp.Bunch(kw)
   smearrad, exhalf = kw.smear_params[ihier]
   cart_resl = cart_extent / (smearrad * 3 - 1 + exhalf)
   ori_nside = xbin_base.ori_nside
   if ori_extent / xbin_base.ori_resl > 1.8:
      ori_nside //= 2
   if smearrad == 0 and exhalf == 0:
      cart_resl = cart_extent

   xbin = rp.xbin.create_Xbin_nside(cart_resl, ori_nside, kw.xbin_max_cart)
   basemap, *_ = rp.motif.create_res_pair_score_map(pairdat, xbin, **kw)
   assert basemap.xbin == xbin

   if smearrad > 0:
      if kw.smear_kernel == "flat":
         kern = []
      if kw.smear_kernel == "x3":  # 1/R**3 uniform in R
         grid_r2 = xbin.grid6.neighbor_sphere_radius_square_cut(smearrad, exhalf)
         kern = 1 - (np.arange(grid_r2 + 1) / grid_r2)**1.5
      smearmap = rp.xbin.smear(xbin, basemap.phmap, radius=smearrad, extrahalf=exhalf, oddlast3=1,
                               sphere=1, kernel=kern)
   else:
      smearmap = basemap.phmap
   sm = rp.Xmap(xbin, smearmap, rehash_bincens=True)
   ori_lever_extent = ori_extent * np.pi / 180 * kw.sampling_lever
   sm.attr.hresl = np.sqrt(cart_extent**2 + ori_lever_extent**2)
   sm.attr.cli_args = kw
   sm.attr.smearrad = smearrad
   sm.attr.exhalf = exhalf
   sm.attr.cart_extent = cart_extent
   sm.attr.ori_extent = ori_extent
   sm.attr.use_ss_key = kw.use_ss_key

   log.info(f"{ihier} {smearrad} {exhalf} cart {cart_extent:6.2f} {cart_resl:6.2f} " +
            f"ori {ori_extent:6.2f} {xbin.ori_resl:6.2f} nsmr {len(smearmap)/1e6:5.1f}M " +
            f"base {len(basemap)/1e3:5.1f}K xpnd {len(smearmap) / len(basemap):7.1f}")

   fname = kw.output_prefix + "%s_%s_%s_p%s_b%s_hier%i_%s_%i_%i.pickle" % (
      kw.score_only_ss, kw.score_only_aa, sstag, _rmzero(f"{kw.min_pair_score}"),
      _rmzero(f"{kw.min_bin_score}"), ihier, "K" + kw.smear_kernel, smearrad, exhalf)
   rp.dump(sm, fname)
   return fname

def _rmzero(a):
   if a[-1] == "0" and "." in a:
      b = a.rstrip("0")
      return b.rstrip(".")
   return a
