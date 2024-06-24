import logging, numpy as np, rpxdock as rp
import willutil as wu

log = logging.getLogger(__name__)

@wu.timed
def filter_redundancy(
   xforms,
   body,
   scores=None,
   categories=None,
   every_nth=1,
   symframes=None,
   debugaggro=False,
   **kw,
):

   # wu.save((xforms, body, scores, categories, every_nth, symframes, kw), 'testdat_C2.pickle')
   # assert 0, 'saving debugging data'

   kw = wu.Bunch(kw, _strict=False)
   if scores is None:
      scores = np.repeat(0, len(xforms))
   if len(scores) == 0: return []

   assert not categories
   if categories is None:
      categories = np.repeat(0, len(scores))

   nclust = kw.max_cluster if kw.max_cluster else int(kw.beam_size) // every_nth
   nclust = min(nclust, len(xforms))
   ibest = np.argsort(-scores)[:nclust]
   xbest = xforms[ibest]
   # ic(ibest.shape, ibest.dtype)
   # ic(categories.shape)
   # categories = categories[ibest]
   # ic(categories.shape)

   iscyclic = False
   if isinstance(symframes, str):
      iscyclic = symframes[0] in 'cC'
      symframes = rp.geom.symframes(symframes, pos=xforms[ibest], **kw)
      # try:
      # symframes = wu.sym.frames(symframes)
      # except (KeyError, ValueError):
      # symframes = None
   if symframes is None:
      symframes = np.array([np.eye(4)])

   if kw.max_bb_redundancy <= 0:
      return ibest

   if xforms.ndim == 3:  # one component
      if symframes.ndim == 3 and len(symframes) > 1 and iscyclic:
         # ic(symframes.shape)
         # x_to_all_frame = np.einsum('dij,sjk,dkl->dsil', wu.hinv(xforms[ibest]), symframes, xforms[ibest])
         # crd = wu.hxform(x_to_all_frame, body.coord[:, 1])
         # crd = np.concatenate([crd[:, 0], crd[:, 1]], axis=1)
         # ic(crd.shape)
         # assert 0
         x_to_2nd_frame = np.einsum('dij,jk,dkl->dil', wu.hinv(xforms[ibest]), symframes[1], xforms[ibest])
         crd = wu.hxform(x_to_2nd_frame, body.cen[::every_nth])
      else:
         crd = wu.hxform(xforms[ibest], body.cen[::every_nth])
   else:
      com0 = wu.hxform(xforms[ibest, 0], body[0].com(), is_points=True, outerprod=True)
      com1 = wu.hxform(xforms[ibest, 1], body[1].com(), is_points=True, outerprod=True)
      bodycrd0 = body[0].cen[::every_nth]
      bodycrd1 = body[1].cen[::every_nth]
      crd0 = wu.hxform(xforms[ibest, 0], bodycrd0, is_points=True, outerprod=False)
      crd1 = wu.hxform(xforms[ibest, 1], bodycrd1, is_points=True, outerprod=False)

      if symframes.ndim == 4:
         # unboundnd syms have one symframe per dock, makes choosing closest subs more involved
         print(f'doing filter_redundancy with sym shape {symframes.shape}')
         com1sym = np.einsum('dsij,dj->sdi', symframes, com1)
         dist = np.linalg.norm((com0 - com1sym), axis=-1)
         isym = np.argmin(dist.reshape(len(dist), -1), axis=0)
         crd1 = np.einsum('dij,drj->dri', symframes[range(len(isym)), isym], crd1)
      else:
         com1sym = wu.hxform(symframes, com1)
         dist = np.linalg.norm(com0 - com1sym, axis=-1)
         isym = np.argmin(dist, axis=0)
         crd1 = wu.hxform(symframes[isym], crd1, is_points=True, outerprod=False)
      crd = np.concatenate([crd0, crd1], axis=1)

   # ic(crd.shape)

   if crd.ndim == 2:
      assert 0, 'why would this happen?'
      return ibest

   ncen = crd.shape[1]
   clustcrd = crd[..., :3].reshape(-1, 3 * ncen)

   # sneaky way to do categories
   # clustcrd += categories[:,None] * 1_000_000

   if debugaggro:
      for i in range(10):
         wu.dumppdb(f'test_cc_in_crd_{i:03}.pdb', crd[i])

   # ic(clustcrd.shape, np.sqrt(ncen), kw.max_bb_redundancy)
   keep, clustid = rp.cluster.cookie_cutter(clustcrd, kw.max_bb_redundancy * np.sqrt(ncen))

   if debugaggro: ic(keep)
   assert len(np.unique(keep)) == len(keep)

   if debugaggro:
      for i in range(10):
         ic(crd.shape)
         ic(crd[keep[i]].shape)
         wu.dumppdb(f'test_cc_crd_keep_{i:03}.pdb', crd[keep[i]])
         # if i in keep:
         # wu.dumppdb(f'test_cc_keep_{i:03}.pdb', crd[i])
         if isinstance(body, rp.Body):
            bodycrd = wu.hxform(xforms[ibest[keep[i]]], body.coord[:, :3])
         else:
            crd0 = wu.hxform(xforms[ibest[keep[i]], 0], body[0].coord[:, :3])
            crd1 = wu.hxform(xforms[ibest[keep[i]], 1], body[1].coord[:, :3])
            bodycrd = np.concatenate([crd0, crd1])
         wu.dumppdb(f'test_cc_keep_bodycoord{i:03}.pdb', bodycrd)

   # log.info(f'filter_redundancy {kw.max_bb_redundancy}A Nmax {nclust} ' + f'Ntotal {len(ibest)} Nkeep {len(keep)}')

   return ibest[keep]

'''
INFO:root:============================= RUNNING dock.py:main =============================
INFO:rpxdock.score.rpxhier:Detected hscore files filetype: ".pickle"
using hscore /rpxdock_files/ilv_h/pdb_res_pair_data_si30_rots_H_ILV_SSindep_p0.5_b1_hier0_Kflat_2_1.pickle
dock_layer
dock_layer (Body(source="inputs/RK1121-BCDA-staple-1_6_F.pdb"), Body(source="inputs/RK718hfuse-01_asu_-0.pdb"))
INFO:rpxdock.search.hierarchical:output/P6_63/RK1121-BCDA-staple-1_6/RK718hfuse-01_asu/P6_63 iresl 0 ntot     720,000 nonzero   499
INFO:rpxdock.search.hierarchical:output/P6_63/RK1121-BCDA-staple-1_6/RK718hfuse-01_asu/P6_63 iresl 1 ntot       7,984 nonzero 3,804
INFO:rpxdock.search.hierarchical:output/P6_63/RK1121-BCDA-staple-1_6/RK718hfuse-01_asu/P6_63 iresl 2 ntot      60,864 nonzero 37,073
INFO:rpxdock.search.hierarchical:output/P6_63/RK1121-BCDA-staple-1_6/RK718hfuse-01_asu/P6_63 iresl 3 ntot     100,000 nonzero 99,125
INFO:rpxdock.search.hierarchical:output/P6_63/RK1121-BCDA-staple-1_6/RK718hfuse-01_asu/P6_63 iresl 4 ntot     100,000 nonzero 99,082
ic| xforms.shape: (100000, 2, 4, 4)
ic| keep: array([    0,     1,     3,    15,    35,    87,    94,   105,   140,
                   180,   185,   190,   360,   579,   916,   978,  1022,  1186,
                  1236,  1294,  1318,  1371,  1379,  1383,  1494,  1502,  1599,
                  1711,  1854,  2352,  2402,  2604,  2758,  2765,  3059,  3086,
                  3174,  3345,  3600,  3783,  3918,  3922,  3952,  4353,  4370,
                  4720,  4721,  4757,  4885,  4896,  4916,  4989,  5408,  5793,
                  6410,  7376,  7574,  7658,  7659,  8448,  8476,  8478,  8479,
                  8870,  8998,  9064,  9065,  9768,  9841, 10276, 10298, 10452,
                 10733, 10919, 11193, 11194, 11212, 11636, 12536, 13975, 14023,
                 14050, 14213, 14973, 15086, 15393, 15447, 15741, 16334, 16335,
                 16467, 16732, 17015, 17660, 17789, 17862, 18144, 18206, 18452,
                 19075, 19125, 19201, 19569, 20112, 20371, 20862, 21573, 22034,
                 22803, 23125, 24185, 24917, 25766, 28177, 28603, 28698, 29017,
                 29366, 30877, 31494, 32445, 33723, 34367, 34527, 34607, 35756,
                 38085, 38110, 38294, 38352, 38591, 39679, 40229, 41678, 41907,
                 41997, 44187, 47263, 47265, 47670, 48261, 50631, 50919, 51499,
                 53519, 56054, 56605, 56922, 57228, 57400, 59345, 60009, 60266,
                 60581, 62973, 65587, 70704, 76189, 76466, 81295, 86558, 91580,
                 94915, 94934, 95746, 99144], dtype=int32)
INFO:rpxdock.search.result:dumping pdb /home/sheffler/project/yangbug/output/P6_63/RK1121-BCDA-staple-1_6/RK718hfuse-01_asu/P6_63_compA_RK1121-BCDA-staple-1_6_F__compB_RK718hfuse-01_asu_-0__top0_0.pdb score 56.8536376953125
INFO:root:dumping /home/sheffler/project/yangbug/output/P6_63/RK1121-BCDA-staple-1_6/RK718hfuse-01_asu/P6_63_compA_RK1121-BCDA-staple-1_6_F__compB_RK718hfuse-01_asu_-0__top0_0.pdb
INFO:rpxdock.search.result:dumping pdb /home/sheffler/project/yangbug/output/P6_63/RK1121-BCDA-staple-1_6/RK718hfuse-01_asu/P6_63_compA_RK1121-BCDA-staple-1_6_F__compB_RK718hfuse-01_asu_-0__top1_1.pdb score 54.76165771484375
INFO:root:dumping /home/sheffler/project/yangbug/output/P6_63/RK1121-BCDA-staple-1_6/RK718hfuse-01_asu/P6_63_compA_RK1121-BCDA-staple-1_6_F__compB_RK718hfuse-01_asu_-0__top1_1.pdb
INFO:rpxdock.search.result:dumping pdb /home/sheffler/project/yangbug/output/P6_63/RK1121-BCDA-staple-1_6/RK718hfuse-01_asu/P6_63_compA_RK1121-BCDA-staple-1_6_F__compB_RK718hfuse-01_asu_-0__top2_2.pdb score 54.670162200927734
INFO:root:dumping /home/sheffler/project/yangbug/output/P6_63/RK1121-BCDA-staple-1_6/RK718hfuse-01_asu/P6_63_compA_RK1121-BCDA-staple-1_6_F__compB_RK718hfuse-01_asu_-0__top2_2.pdb
INFO:rpxdock.search.result:dumping pdb /home/sheffler/project/yangbug/output/P6_63/RK1121-BCDA-staple-1_6/RK718hfuse-01_asu/P6_63_compA_RK1121-BCDA-staple-1_6_F__compB_RK718hfuse-01_asu_-0__top3_3.pdb score 52.80339050292969
INFO:root:dumping /home/sheffler/project/yangbug/output/P6_63/RK1121-BCDA-staple-1_6/RK718hfuse-01_asu/P6_63_compA_RK1121-BCDA-staple-1_6_F__compB_RK718hfuse-01_asu_-0__top3_3.pdb
INFO:rpxdock.search.result:dumping pdb /home/sheffler/project/yangbug/output/P6_63/RK1121-BCDA-staple-1_6/RK718hfuse-01_asu/P6_63_compA_RK1121-BCDA-staple-1_6_F__compB_RK718hfuse-01_asu_-0__top4_4.pdb score 51.2770881652832
INFO:root:dumping /home/sheffler/project/yangbug/output/P6_63/RK1121-BCDA-staple-1_6/RK718hfuse-01_asu/P6_63_compA_RK1121-BCDA-staple-1_6_F__compB_RK718hfuse-01_asu_-0__top4_4.pdb
INFO:rpxdock.search.result:dumping pdb /home/sheffler/project/yangbug/output/P6_63/RK1121-BCDA-staple-1_6/RK718hfuse-01_asu/P6_63_compA_RK1121-BCDA-staple-1_6_F__compB_RK718hfuse-01_asu_-0__top5_5.pdb score 49.51369094848633
INFO:root:dumping /home/sheffler/project/yangbug/output/P6_63/RK1121-BCDA-staple-1_6/RK718hfuse-01_asu/P6_63_compA_RK1121-BCDA-staple-1_6_F__compB_RK718hfuse-01_asu_-0__top5_5.pdb
INFO:rpxdock.search.result:dumping pdb /home/sheffler/project/yangbug/output/P6_63/RK1121-BCDA-staple-1_6/RK718hfuse-01_asu/P6_63_compA_RK1121-BCDA-staple-1_6_F__compB_RK718hfuse-01_asu_-0__top6_6.pdb score 49.34780502319336
INFO:root:dumping /home/sheffler/project/yangbug/output/P6_63/RK1121-BCDA-staple-1_6/RK718hfuse-01_asu/P6_63_compA_RK1121-BCDA-staple-1_6_F__compB_RK718hfuse-01_asu_-0__top6_6.pdb
INFO:rpxdock.search.result:dumping pdb /home/sheffler/project/yangbug/output/P6_63/RK1121-BCDA-staple-1_6/RK718hfuse-01_asu/P6_63_compA_RK1121-BCDA-staple-1_6_F__compB_RK718hfuse-01_asu_-0__top7_7.pdb score 48.84437942504883
INFO:root:dumping /home/sheffler/project/yangbug/output/P6_63/RK1121-BCDA-staple-1_6/RK718hfuse-01_asu/P6_63_compA_RK1121-BCDA-staple-1_6_F__compB_RK718hfuse-01_asu_-0__top7_7.pdb
INFO:rpxdock.search.result:dumping pdb /home/sheffler/project/yangbug/output/P6_63/RK1121-BCDA-staple-1_6/RK718hfuse-01_asu/P6_63_compA_RK1121-BCDA-staple-1_6_F__compB_RK718hfuse-01_asu_-0__top8_8.pdb score 47.968544006347656
INFO:root:dumping /home/sheffler/project/yangbug/output/P6_63/RK1121-BCDA-staple-1_6/RK718hfuse-01_asu/P6_63_compA_RK1121-BCDA-staple-1_6_F__compB_RK718hfuse-01_asu_-0__top8_8.pdb
INFO:rpxdock.search.result:dumping pdb /home/sheffler/project/yangbug/output/P6_63/RK1121-BCDA-staple-1_6/RK718hfuse-01_asu/P6_63_compA_RK1121-BCDA-staple-1_6_F__compB_RK718hfuse-01_asu_-0__top9_9.pdb score 46.908180236816406
INFO:root:dumping /home/sheffler/project/yangbug/output/P6_63/RK1121-BCDA-staple-1_6/RK718hfuse-01_asu/P6_63_compA_RK1121-BCDA-staple-1_6_F__compB_RK718hfuse-01_asu_-0__top9_9.pdb
INFO:rpxdock.search.result:dumping pdb /home/sheffler/project/yangbug/output/P6_63/RK1121-BCDA-staple-1_6/RK718hfuse-01_asu/P6_63_compA_RK1121-BCDA-staple-1_6_F__compB_RK718hfuse-01_asu_-0__job0_top0_0.pdb score 56.8536376953125
INFO:root:dumping /home/sheffler/project/yangbug/output/P6_63/RK1121-BCDA-staple-1_6/RK718hfuse-01_asu/P6_63_compA_RK1121-BCDA-staple-1_6_F__compB_RK718hfuse-01_asu_-0__job0_top0_0.pdb
INFO:root:saved result to output/P6_63/RK1121-BCDA-staple-1_6/RK718hfuse-01_asu/P6_63_Result.pickle
INFO:root:Result with data = <xarray.Dataset>
  Dimensions:   (model: 166, comp: 2, hrow: 4, hcol: 4)
  Dimensions without coordinates: model, comp, hrow, hcol
  Data variables:
      scores    (model) float32 56.85 54.76 54.67 52.8 ... 9.392 9.382 8.688 0.0
      xforms    (model, comp, hrow, hcol) float32 0.5047 -0.8633 0.0 ... 0.0 1.0
      rpx       (model) float32 56.39 54.22 54.18 52.35 ... 9.182 9.182 8.518 0.0
      ncontact  (model) float32 46.0 54.0 49.0 45.0 45.0 ... 21.0 20.0 17.0 0.0
      disp0     (model) float64 0.0 0.0 0.0 0.0 0.0 0.0 ... 0.0 0.0 0.0 0.0 0.0
      angle0    (model) float64 59.69 29.06 22.81 17.81 ... 59.69 35.94 54.69
      disp1     (model) float64 4.062 4.062 4.062 11.56 ... 14.69 28.44 9.062
      angle1    (model) float64 59.69 59.69 77.81 19.69 ... 59.69 0.3125 79.69
      ijob      (model) int64 0 0 0 0 0 0 0 0 0 0 0 0 ... 0 0 0 0 0 0 0 0 0 0 0 0
  Attributes:
      dockinfo:       [{'arg': Bunch(inputs=[['inputs/RK1121-BCDA-staple-1_6_F....
      ttotal:         44.86083151300045
      output_prefix:  output/P6_63/RK1121-BCDA-staple-1_6/RK718hfuse-01_asu/P6_63
      output_body:    all
      sym:            P6_63
INFO:root:<xarray.DataArray 'scores' (model: 166)>
array([  0,   1,   2,   3,   4,   5,   6,   7,   8,   9,  10,  11,  12,
        13,  14,  15,  16,  17,  18,  19,  20,  21,  22,  23,  24,  25,
        26,  27,  28,  29,  30,  31,  32,  33,  34,  35,  36,  37,  38,
        39,  40,  41,  42,  43,  44,  45,  46,  47,  48,  49,  50,  51,
        52,  53,  54,  55,  56,  57,  58,  59,  60,  62,  61,  63,  64,
        65,  66,  67,  68,  69,  70,  71,  72,  73,  74,  75,  76,  77,
        78,  79,  80,  81,  82,  83,  84,  85,  86,  87,  88,  89,  90,
        91,  92,  93,  94,  95,  96,  97,  98,  99, 100, 101, 102, 103,
       104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116,
       117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129,
       130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142,
       143, 144, 145, 146, 147, 148, 149, 150, 151, 152, 153, 154, 155,
       156, 157, 158, 159, 160, 161, 162, 163, 164, 165])
Dimensions without coordinates: model
INFO:root:saved summary to output/P6_63/RK1121-BCDA-staple-1_6/RK718hfuse-01_asu/P6_63_Summary.txt
INFO:root:DONE
doing filter_redundancy with sym shape (100000, 18, 4, 4)

'''