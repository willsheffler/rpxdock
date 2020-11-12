import numpy as np, logging
from rpxdock.sampling.xform_hierarchy import XformHier_f4

log = logging.getLogger(__name__)

def xform_hier_guess_sampling_covrads(
   hierarchy_depth,
   sampling_lever,
   base_sample_resl,
   base_cart_resl,
   base_ori_resl,
   xhier_cart_fudge_factor,
   xhier_ori_fudge_factor,
   **kw,
):
   resl = base_sample_resl * 2**(hierarchy_depth - 1)
   log.debug("low tot resl {resl} {base_sample_resl}")

   cart_resl = resl / np.sqrt(2)
   ori_resl = resl / np.sqrt(2) / sampling_lever * 180 / np.pi
   cart_resl_fudge = cart_resl * xhier_cart_fudge_factor
   ori_resl_fudge = ori_resl * xhier_ori_fudge_factor

   xh = XformHier_f4([0, 0, 0], [1, 1, 1], [1, 1, 1], ori_resl_fudge)
   log.debug(
      f"lever {sampling_lever:7.3f} resoultion {resl:7.3f} cart {cart_resl:7.3f}" +
      f"ori {xh.ori_resl:7.3f} ori actual/request {xh.ori_resl / ori_resl:7.3f}", )
   ori_resl_fudge = xh.ori_resl
   ori_resl = ori_resl_fudge / xhier_ori_fudge_factor
   h_resl = list()
   for i in range(hierarchy_depth):
      h_cart_resl = cart_resl * 0.5**i
      h_ori_resl = ori_resl * 0.5**i
      hb_cart_resl = h_cart_resl + base_cart_resl
      hb_ori_resl = h_ori_resl + base_ori_resl
      h_cart_resl_fudge = h_cart_resl * xhier_cart_fudge_factor
      h_ori_resl_fudge = h_ori_resl * xhier_ori_fudge_factor
      log.debug(f"stage {i} samp: {h_cart_resl:6.3f} {h_ori_resl:6.3f}" +
                f"score: {hb_cart_resl:6.3f} {hb_ori_resl:6.3f}" +
                f"fugded: {h_cart_resl_fudge:6.3f} {h_ori_resl_fudge:6.3f}")
      h_resl.append((hb_cart_resl, hb_ori_resl))

   cart_side = cart_resl * 2.0 / np.sqrt(3)
   nsamp_per_nm3 = int(xh.size(0) * ((1000 / cart_side) / 100)**3)

   return h_resl, (cart_resl_fudge, ori_resl_fudge), nsamp_per_nm3
