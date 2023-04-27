import os, logging, glob, itertools as it
import numpy as np
import rpxdock as rp
from rpxdock.xbin import xbin_util as xu
from rpxdock.score import score_functions as sfx
import willutil as wu
#from icecream import ic
#ic.configureOutput(includeContext=True)

log = logging.getLogger(__name__)
"""
RpxHier holds score information at each level of searching / scoring 
Grid search just uses the last/finest scorefunction 
"""

class RpxHier:
   def __init__(self, files, max_pair_dist=8.0, **kw):
      kw = wu.Bunch(kw, _strict=False)
      hscore_stuff = read_hscore_files(files, **kw)

      self.use_ss = hscore_stuff.use_ss
      self.base = hscore_stuff.base
      self.hier = hscore_stuff.hier
      self.resl = hscore_stuff.resl

      self.actual_nresl = len(self.hier)
      for i in range(10):
         self.hier.append(self.hier[-1])
      self.cart_extent = [h.attr.cart_extent for h in self.hier]
      self.ori_extent = [h.attr.ori_extent for h in self.hier]
      self.max_pair_dist = [max_pair_dist + h.attr.cart_extent for h in self.hier]
      self.map_pairs_multipos = xu.ssmap_pairs_multipos if self.use_ss else xu.map_pairs_multipos
      self.map_pairs = xu.ssmap_of_selected_pairs if self.use_ss else xu.map_of_selected_pairs
      self.score_only_sspair = kw.get('score_only_sspair')
      self.function = kw.function

   def __len__(self):
      return len(self.hier)

   def hier_mindis(self, iresl):
      return [1.5, 2.0, 2.75, 3.25, 3.5][iresl]

   def score(self, body1, body2, wts, iresl=-1, *bounds):
      return self.scorepos(body1, body2, body1.pos, body2.pos, iresl, *bounds, wts=wts)

   def score_matrix_intra(self, body, wts, iresl=-1):
      pairs, lbub = rp.bvh.bvh_collect_pairs_vec(body.bvh_cen, body.bvh_cen, np.eye(4), np.eye(4),
                                                 self.max_pair_dist[iresl])
      pairs = body.filter_pairs(pairs, self.score_only_sspair)
      assert len(lbub) == 1
      xmap = self.hier[iresl]
      ssstub = body.ssid, body.ssid, body.stub, body.stub
      if not self.use_ss: ssstub = ssstub[2:]
      pscore = self.map_pairs(xmap.xbin, xmap.phmap, pairs, *ssstub)
      m = np.zeros((len(body), ) * 2, dtype='f4')
      m[pairs[:, 0], pairs[:, 1]] += pscore
      m[pairs[:, 1], pairs[:, 0]] += pscore
      m /= 2
      return m

   def score_matrix_inter(self, bodyA, bodyB, wts, symframes=[np.eye(4)], iresl=-1, **kw):
      m = np.zeros((len(bodyA), len(bodyB)), dtype='f4')
      symframes = [np.eye(4, dtype='f4')]
      for xsym in symframes[1:]:
         #.astype('f4')
         pairs, lbub = rp.bvh.bvh_collect_pairs_vec(
            bodyA.bvh_cen,
            bodyB.bvh_cen,
            bodyA.pos,
            xsym @ bodyB.pos,
            self.max_pair_dist[iresl],
         )
         assert len(lbub) == 1
         pairs = bodyA.filter_pairs(pairs, self.score_only_sspair, other=bodyB, **kw)
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

   def scorepos(
         self,
         body1,
         body2,
         pos1,
         pos2,
         iresl=-1,
         bounds=(),
         residue_summary=np.mean,  # TODO hook up to options to select
         termini_max_dist=9e9,
         **kw,
   ):

      kw = wu.Bunch(kw, _strict=False)
      origshape = pos1.shape[:-2] if pos1.ndim > pos2.ndim else pos2.shape[:-2]
      # ic(pos1.shape)
      # ic(pos2.shape)
      # assert pos1.shape[:-2] == pos2.shape[:-2]
      pos1, pos2 = pos1.reshape(-1, 4, 4), pos2.reshape(-1, 4, 4)

      # docks we don't need to score because we already know they are bad
      excluded = np.zeros(max(len(pos1), len(pos2)), dtype='?')
      if termini_max_dist < 9e8 and not bounds:
         tmindis = np.ones(len(pos1)) * 9e9
         for term1, term2 in it.chain(it.product(body1.nterms, body2.cterms), it.product(body1.cterms, body2.nterms)):
            tpos1 = pos1 @ term1
            tpos2 = pos2 @ term2
            d = np.linalg.norm(tpos1 - tpos2, axis=-1)
            tmindis = np.minimum(tmindis, d)
            # ic(np.min(tmindis))
            # ic(tpos1.shape, tpos2.shape, d.shape)
         excluded = tmindis > termini_max_dist + self.cart_extent[iresl]
         pos1 = pos1[~excluded]
         pos2 = pos2[~excluded]

      bounds = list(bounds)
      if len(bounds) > 2 and (bounds[2] is None or bounds[2] < 0):
         bounds[2] = body1.asym_body.nres
      if len(bounds) > 5 and (bounds[5] is None or bounds[5] < 0):
         bounds[5] = body2.asym_body.nres

      # calling bvh c++ function that will look at pair of (arrays of) positions, scores pairs that are in contact (ID from maxpair distance)
      # lbub: len pos1
      pairs, lbub = rp.bvh.bvh_collect_pairs_range_vec(
         body1.bvh_cen,
         body2.bvh_cen,
         pos1,
         pos2,
         self.max_pair_dist[iresl],
         *bounds,
      )
      if len(pairs) == 0:
         return np.zeros(len(pos1))

      contains_res = list()
      if hasattr(body1, 'required_res_sets') and body1.required_res_sets:
         res0 = np.ascontiguousarray(pairs[:, 0])
         # ic(res0.shape)
         hasit0 = [s.has(res0) for s in body1.required_res_sets]
         contains_res.extend(hasit0)

      if hasattr(body2, 'required_res_sets') and body2.required_res_sets:
         res1 = np.ascontiguousarray(pairs[:, 1])
         hasit1 = [s.has(res1) for s in body2.required_res_sets]
         contains_res.extend(hasit1)

      ok = np.ones(len(lbub), dtype='?')
      if contains_res:
         for i, (lb, ub) in enumerate(lbub):
            ncontains = [np.sum(cr[lb:ub]) for cr in contains_res]
            ok[i] = all([10 < n for n in ncontains])
            # if i < 3:
            # ic(kw.ibody, i, ncontains)

      # "remove" all docks that don't have required set of contacts
      lbub[~ok, 1] = lbub[~ok, 0]

      #

      # pairs, lbub = body1.filter_pairs(pairs, self.score_only_sspair, other=body2, lbub=lbub)

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

      #TODO: Figure out if this should be handled in the score functions below.
      if 'wts' in kw and kw.wts.rpx == 0:
         return kw.wts.ncontact * (lbub[:, 1] - lbub[:, 0])
         # option to score based on ncontacts only

      xbin = self.hier[iresl].xbin
      phmap = self.hier[iresl].phmap
      ssstub = body1.ssid, body2.ssid, body1.stub, body2.stub
      ssstub = ssstub if self.use_ss else ssstub[2:]

      # hashtable of scores for each pair of res in contact in each dock
      pscore = self.map_pairs_multipos(
         xbin,
         phmap,
         pairs,
         *ssstub,
         # body1.ssid, body2.ssid, body1.stub, body2.stub,
         lbub,
         pos1,
         pos2,
         incomplete_ok=True,
      )

      # summarize pscores for a dock
      lbub1, lbub2, idx1, idx2, ressc1, ressc2 = rp.motif.marginal_max_score(
         lbub,
         pairs,
         pscore,
      )

      score_functions = {
         "fun2": sfx.score_fun2,
         "lin": sfx.lin,
         "exp": sfx.exp,
         "mean": sfx.mean,
         "median": sfx.median,
         "stnd": sfx.stnd,
         "sasa_priority": sfx.sasa_priority
      }
      score_fx = score_functions.get(self.function)

      if score_fx:
         scores = score_fx(pos1, pos2, lbub, lbub1, lbub2, ressc1, ressc2, pairs=pairs, wts=kw.wts, iresl=iresl)
      else:
         logging.debug(f"Failed to find score function {self.function}, falling back to 'stnd'")
         scores = score_functions["stnd"](pos1, pos2, lbub, lbub1, lbub2, ressc1, ressc2, wts=kw.wts)

      scores[~ok] = 0

      # insert non-excluded scores into full score array
      all_scores = np.zeros(len(excluded))
      all_scores[~excluded] = scores
      all_scores = all_scores.reshape(origshape)

      return all_scores

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

def get_hscore_file_names(alias, hscore_data_dir):
   # try:
   picklepattern1 = os.path.join(hscore_data_dir, alias, '*.pickle')
   picklefiles1 = sorted(glob.glob(picklepattern1))
   picklepattern2 = os.path.join(hscore_data_dir, alias, '*.pickle.bz2')
   picklefiles2 = sorted(glob.glob(picklepattern2))
   picklefiles = picklefiles1 or picklefiles2
   xmappattern = os.path.join(hscore_data_dir, alias, '*.txz')
   txzfiles = sorted(glob.glob(xmappattern))
   fnames = txzfiles
   # ic(picklepattern1)
   # ic(picklefiles)
   # print(alias)
   # print(hscore_data_dir)
   # assert 0
   if len(picklefiles):
      assert len(txzfiles) in (0, len(txzfiles))
      fnames = picklefiles
      # for f in fnames:
      # print(' ', f)
      # ic(sum([s.count('base') for s in fnames]))
      assert sum([s.count('base') for s in fnames]) < 2
      # assert sum([s.count('.rpx.pickle') for s in fnames]) == 1
   else:
      print(
         'WARNING: using slower, portable tarball format. generate pickle files with --generate_hscore_pickle_files and place with original .txz files for faster operation!'
      )
      '''
      ic(alias)
      ic(hscore_data_dir)
      ic(picklepattern)
      ic(picklefiles1)
      ic(picklefiles2)
      ic(picklefiles)
      '''
      # assert 0
   if not fnames:
      raise ValueError(f'not hscore files found for "{alias}" in "{hscore_data_dir}"')

   for filetype in '.txz .pickle .pickle.gz .pickle.bz2 .pickle.zip'.split():
      if fnames[0].endswith(filetype):
         log.info(f'Detected hscore files filetype: "{filetype}"')
         for f in fnames:
            assert f.endswith(filetype), 'inconsistent hscore filetypes, all must be same (.gz, .bz2, .pickle, etc)'
   return fnames

# except:
#    pass
# raise ValueError(f'hscore datadir {hscore_data_dir} or alias {alias} invalid')
def read_hscore_files(
   files,
   hscore_data_dir='',
   generate_hscore_pickle_files=False,
   **kw,
):
   toreturn = wu.Bunch()

   if isinstance(files, str): files = [files]
   if len(files) == 0: raise ValueError('RpxHier given no datafiles')
   if len(files) == 1: files = get_hscore_file_names(files[0], hscore_data_dir)
   if len(files) > 8:
      for f in files:
         log.error(f)
      raise ValueError('too many hscore_files (?)')
   assert files

   if all(isinstance(f, str) for f in files):
      if "_SSindep_" in files[0]:
         assert all("_SSindep_" in f for f in files)
         toreturn.use_ss = False
      else:
         assert all("_SSdep_" in f for f in files)
         toreturn.use_ss = True
      data = rp.util.load_threads(files, len(files))
      if "base" in files[0]:
         toreturn.base = data[0]
         toreturn.hier = data[1:]
      else:
         toreturn.base = None
         toreturn.hier = data

      for h in toreturn.hier:
         h.attr = wu.Bunch(h.attr)
      toreturn.resl = list(h.attr.cart_extent for h in toreturn.hier)

      if generate_hscore_pickle_files and 'pickle' not in files[0].split('.')[-2:]:
         for d, f in zip(data, files):
            print(f'converting {f} to pickle')
            newf = os.path.basename(f) + '.pickle'
            print('saving', newf)
            rp.dump(d, newf)
            # print('gzip', newf)
            # os.system(f'gzip -f {newf}')
            print()
         print('Faster but non-portable .pickle cache files generated from .txz files')
         print('move these into same directory as original .txz files and rpxdock will use them')
         import sys
         sys.exit()

   elif (isinstance(files[0], rp.ResPairScore) and all(isinstance(f, rp.Xmap) for f in files[1:])):
      toreturn.base = files[0]
      toreturn.hier = list(files[1:])
      toreturn.use_ss = toreturn.base.attr.opts.use_ss_key
      assert all(toreturn.use_ss == h.attr.cli_args.use_ss_key for h in toreturn.hier)

   else:
      raise ValueError('RpxHier expects filenames or ResPairScore+[Xmap*]')
   # append extra copies of highest resl score to use for higher res search steps

   return toreturn
