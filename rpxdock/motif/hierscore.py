import os, _pickle, threading, logging, glob, numpy as np
from concurrent.futures import ThreadPoolExecutor
import rpxdock as sd
from rpxdock.rotamer import get_rotamer_space
from rpxdock.util import Bunch
from rpxdock.sampling import xform_hier_guess_sampling_covrads
from rpxdock.xbin import smear, Xbin
from rpxdock.xbin import xbin_util as xu
from rpxdock.util import load, dump, load_threads, InProcessExecutor
from rpxdock.motif import Xmap, ResPairScore, marginal_max_score
from rpxdock.bvh import bvh_collect_pairs_range_vec, bvh_collect_pairs_vec

log = logging.getLogger(__name__)

class HierScore:
   def __init__(self, files, max_pair_dist=8.0, hscore_data_dir=None, **kw):
      arg = sd.Bunch(kw)
      if isinstance(files, str): files = [files]
      if len(files) is 0: raise ValueError('HierScore given no datafiles')
      if len(files) is 1: files = _check_hscore_files_aliases(files[0], hscore_data_dir)
      if len(files) > 8:
         for f in files:
            log.error(f)
         raise ValueError('too many hscore_files (?)')
      if all(isinstance(f, str) for f in files):
         if "_SSindep_" in files[0]:
            assert all("_SSindep_" in f for f in files)
            self.use_ss = False
         else:
            assert all("_SSdep_" in f for f in files)
            self.use_ss = True
         assert "base" in files[0]
         data = load_threads(files, len(files))
         self.base = data[0]
         self.hier = data[1:]
         self.resl = list(h.attr.cart_extent for h in self.hier)
      elif isinstance(files[0], ResPairScore) and all(isinstance(f, Xmap) for f in files[1:]):
         self.base = files[0]
         self.hier = list(files[1:])
         self.use_ss = self.base.attr.opts.use_ss_key
         assert all(self.use_ss == h.attr.cli_args.use_ss_key for h in self.hier)
      else:
         raise ValueError('HierScore expects filenames or ResPairScore+[Xmap*]')
      self.max_pair_dist = [max_pair_dist + h.attr.cart_extent for h in self.hier]
      self.map_pairs_multipos = xu.ssmap_pairs_multipos if self.use_ss else xu.map_pairs_multipos
      self.map_pairs = xu.ssmap_of_selected_pairs if self.use_ss else xu.map_of_selected_pairs
      self.score_only_sspair = arg.score_only_sspair

   def __len__(self):
      return len(self.hier)

   def hier_mindis(self, iresl):
      return [1.5, 2.0, 2.75, 3.25, 3.5][iresl]

   def score(self, body1, body2, wts, iresl=-1, *bounds):
      return self.scorepos(body1, body2, body1.pos, body2.pos, iresl, *bounds, wts=wts)

   def score_matrix_intra(self, body, wts, iresl=-1):
      pairs, lbub = bvh_collect_pairs_vec(body.bvh_cen, body.bvh_cen, np.eye(4), np.eye(4),
                                          self.max_pair_dist[iresl])
      pairs = body.filter_pairs(pairs, self.score_only_sspair)
      assert len(lbub) is 1
      xmap = self.hier[iresl]
      ssstub = body.ssid, body.ssid, body.stub, body.stub
      if not self.use_ss: ssstub = ssstub[2:]
      pscore = self.map_pairs(xmap.xbin, xmap.phmap, pairs, *ssstub)
      m = np.zeros((len(body), ) * 2, dtype='f4')
      m[pairs[:, 0], pairs[:, 1]] += pscore
      m[pairs[:, 1], pairs[:, 0]] += pscore
      m /= 2
      return m

   def score_matrix_inter(self, bodyA, bodyB, wts, symframes=[np.eye(4)], iresl=-1):
      m = np.zeros((len(bodyA), len(bodyB)), dtype='f4')
      for xsym in symframes.astype('f4'):
         pairs, lbub = bvh_collect_pairs_vec(bodyA.bvh_cen, bodyB.bvh_cen, bodyA.pos,
                                             xsym @ bodyB.pos, self.max_pair_dist[iresl])
         assert len(lbub) is 1
         pairs = bodyA.filter_pairs(pairs, self.score_only_sspair, other=bodyB)
         xmap = self.hier[iresl]
         ssstub = bodyA.ssid, bodyB.ssid, bodyA.pos @ bodyA.stub, xsym @ bodyB.pos @ bodyB.stub
         if not self.use_ss: ssstub = ssstub[2:]

         pscore = self.map_pairs(xmap.xbin, xmap.phmap, pairs, *ssstub)
         m[pairs[:, 0], pairs[:, 1]] += pscore

      return m

#  m.def("map_of_selected_pairs", &map_of_selected_pairs_onearray<K, F, double>,
#        "xbin"_a, "phmap"_a, "idx"_c, "xform1"_c, "xform2"_c, "pos1"_a = eye4,
#        "pos2"_a = eye4);
#  m.def("map_of_selected_pairs",
#        &map_of_selected_pairs_onearray_same<K, F, double>, "xbin"_a, "phmap"_a,
#        "idx"_c, "xform"_c, "pos1"_a = eye4, "pos2"_a = eye4);
#
#  m.def("ssmap_of_selected_pairs",
#        &ssmap_of_selected_pairs_onearray<K, F, double>, "xbin"_a, "phmap"_a,
#        "idx"_c, "ss1"_c, "ss2"_c, "xform1"_c, "xform2"_c, "pos1"_a = eye4,
#        "pos2"_a = eye4);
#  m.def("ssmap_of_selected_pairs",
#        &ssmap_of_selected_pairs_onearray_same<K, F, double>, "xbin"_a,
#        "phmap"_a, "idx"_c, "ss"_c, "xform"_c, "pos1"_a = eye4,
#        "pos2"_a = eye4);

   def scorepos(self, body1, body2, pos1, pos2, iresl=-1, bounds=(), **kw):
      arg = Bunch(kw)
      pos1, pos2 = pos1.reshape(-1, 4, 4), pos2.reshape(-1, 4, 4)
      # if not bounds:
      # bounds = [-2e9], [2e9], nsym[0], [-2e9], [2e9], nsym[1]
      # if len(bounds) is 2:
      # bounds += nsym[1],
      # if len(bounds) is 3:
      # bounds += [-2e9], [2e9], 1
      bounds = list(bounds)
      if len(bounds) > 2 and (bounds[2] is None or bounds[2] < 0):
         bounds[2] = body1.asym_body.nres
      if len(bounds) > 5 and (bounds[5] is None or bounds[5] < 0):
         bounds[5] = body2.asym_body.nres

      # print('nres asym', body1.asym_body.nres, body2.asym_body.nres)
      # print(bounds[2], bounds[5])
      pairs, lbub = bvh_collect_pairs_range_vec(body1.bvh_cen, body2.bvh_cen, pos1, pos2,
                                                self.max_pair_dist[iresl], *bounds)

      if bounds: assert len(bounds[0]) in (1, len(lbub))
      # if len(bounds[0]) > 1:
      #    print(len(lbub), len(bounds[0]))

      #    # print(lbub)
      #    for i, (lb, ub) in enumerate(lbub):
      #       asym_res1 = pairs[lb:ub, 0] % body1.asym_body.nres
      #       asym_res2 = pairs[lb:ub, 1] % body2.asym_body.nres
      #       print(i, f'{np.min(asym_res1)}-{np.max(asym_res1)}', bounds[0][i], bounds[1][i])
      #       print(i, f'{np.min(asym_res2)}-{np.max(asym_res2)}', bounds[3][i], bounds[4][i])
      #       assert np.all(asym_res1 >= bounds[0][i])
      #       assert np.all(asym_res1 <= bounds[1][i])
      #       assert np.all(asym_res2 >= bounds[3][i])
      #       assert np.all(asym_res2 <= bounds[4][i])

      if arg.wts.rpx == 0:
         return arg.wts.ncontact * (lbub[:, 1] - lbub[:, 0])

      xbin = self.hier[iresl].xbin
      phmap = self.hier[iresl].phmap
      ssstub = body1.ssid, body2.ssid, body1.stub, body2.stub
      ssstub = ssstub if self.use_ss else ssstub[2:]
      pscore = self.map_pairs_multipos(xbin, phmap, pairs, *ssstub, lbub, pos1, pos2)

      lbub1, lbub2, idx1, idx2, res1, res2 = marginal_max_score(lbub, pairs, pscore)

      scores = np.zeros(max(len(pos1), len(pos2)))
      for i, (lb, ub) in enumerate(lbub):
         side1 = np.sum(res1[lbub1[i, 0]:lbub1[i, 1]])
         side2 = np.sum(res2[lbub2[i, 0]:lbub2[i, 1]])
         mscore = side1 + side2
         # mscore = np.sum(pscore[lb:ub])
         # mscore = np.log(np.sum(np.exp(pscore[lb:ub])))
         scores[i] = arg.wts.rpx * mscore + arg.wts.ncontact * (ub - lb)

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
   xbin = Xbin(cart_resl, ori_resl, max_cart)
   if xbin.ori_nside % 2 != 0:
      xbin = sd.xbin.create_Xbin_nside(cart_resl, xbin.ori_nside + 1, max_cart)
   return xbin

def make_and_dump_hier_score_tables(rp, **kw):
   arg = Bunch(kw)
   fnames = list()

   resls, xhresl, nbase_nm3 = xform_hier_guess_sampling_covrads(**arg)
   # xbin_base = Xbin(arg.base_cart_resl, ORI_RESL, arg.xbin_max_cart)
   xbin_base = create_xbin_even_nside(arg.base_cart_resl, arg.base_ori_resl, arg.xbin_max_cart)
   rps = sd.motif.create_res_pair_score(rp, xbin_base, **arg)
   rps.attr.opts = arg
   rps.attr.nbase_nm3 = nbase_nm3
   rps.attr.xhresl = xhresl

   log.debug(f'base_cart_resl {arg.base_cart_resl} resls[-1][0] {resls[-1][0]}')
   sstag = "SSdep" if arg.use_ss_key else "SSindep"
   ftup = arg.score_only_ss, arg.score_only_aa, sstag, _rmzero(f"{arg.min_pair_score}"), _rmzero(
      f"{arg.min_bin_score}")
   fnames.append(arg.output_prefix + "%s_%s_%s_p%s_b%s_base.pickle" % ftup)
   dump(rps, fnames[-1])

   if len(arg.smear_params) == 1:
      arg.smear_params = arg.smear_params * len(resls)
   assert len(arg.smear_params) == len(resls)

   exe = InProcessExecutor()
   if arg.nthread > 1:
      exe = ThreadPoolExecutor(arg.nthread)
   if arg.nthread < 0:
      exe = ThreadPoolExecutor(arg.ncpu)
   futures = list()
   for ihier, (cart_extent, ori_extent) in enumerate(resls):
      if arg.only_do_hier >= 0 and arg.only_do_hier != ihier:
         continue
      f = futures.append(
         exe.submit(make_and_dump_hier_score_tables_one, rp, ihier, xbin_base, cart_extent,
                    ori_extent, sstag, **arg))
   fnames.extend(f.result() for f in futures)
   return [f.replace('//', '/') for f in fnames]

def make_and_dump_hier_score_tables_one(rp, ihier, xbin_base, cart_extent, ori_extent, sstag,
                                        **kw):
   arg = Bunch(kw)
   smearrad, exhalf = arg.smear_params[ihier]
   cart_resl = cart_extent / (smearrad * 3 - 1 + exhalf)
   ori_nside = xbin_base.ori_nside
   if ori_extent / xbin_base.ori_resl > 1.8:
      ori_nside //= 2
   if smearrad == 0 and exhalf == 0:
      cart_resl = cart_extent

   xbin = sd.xbin.create_Xbin_nside(cart_resl, ori_nside, arg.xbin_max_cart)
   basemap, *_ = sd.motif.create_res_pair_score_map(rp, xbin, **arg)
   assert basemap.xbin == xbin

   if smearrad > 0:
      if arg.smear_kernel == "flat":
         kern = []
      if arg.smear_kernel == "x3":  # 1/R**3 uniform in R
         grid_r2 = xbin.grid6.neighbor_sphere_radius_square_cut(smearrad, exhalf)
         kern = 1 - (np.arange(grid_r2 + 1) / grid_r2)**1.5
      smearmap = smear(xbin, basemap.phmap, radius=smearrad, extrahalf=exhalf, oddlast3=1,
                       sphere=1, kernel=kern)
   else:
      smearmap = basemap.phmap
   sm = sd.motif.Xmap(xbin, smearmap, rehash_bincens=True)
   ori_lever_extent = ori_extent * np.pi / 180 * arg.sampling_lever
   sm.attr.hresl = np.sqrt(cart_extent**2 + ori_lever_extent**2)
   sm.attr.cli_args = arg
   sm.attr.smearrad = smearrad
   sm.attr.exhalf = exhalf
   sm.attr.cart_extent = cart_extent
   sm.attr.ori_extent = ori_extent
   sm.attr.use_ss_key = arg.use_ss_key

   log.info(f"{ihier} {smearrad} {exhalf} cart {cart_extent:6.2f} {cart_resl:6.2f}" +
            f"ori {ori_extent:6.2f} {xbin.ori_resl:6.2f} nsmr {len(smearmap)/1e6:5.1f}M" +
            f"base {len(basemap)/1e3:5.1f}K xpnd {len(smearmap) / len(basemap):7.1f}")

   fname = arg.output_prefix + "%s_%s_%s_p%s_b%s_hier%i_%s_%i_%i.pickle" % (
      arg.score_only_ss, arg.score_only_aa, sstag, _rmzero(f"{arg.min_pair_score}"),
      _rmzero(f"{arg.min_bin_score}"), ihier, "K" + arg.smear_kernel, smearrad, exhalf)
   dump(sm, fname)
   return fname

def _rmzero(a):
   if a[-1] == "0" and "." in a:
      b = a.rstrip("0")
      return b.rstrip(".")
   return a

def _check_hscore_files_aliases(alias, hscore_data_dir):
   try:
      pattern = os.path.join(hscore_data_dir, alias, '*.pickle')
      g = sorted(glob.glob(pattern))
      if len(g) > 0:
         return g
   except:
      pass
   raise ValueError(f'hscore datadir {hscore_data_dir} or alias {alias} invalid')
