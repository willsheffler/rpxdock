import argparse, sys
import numpy as np
import xarray as xr
import rpxdock as rp
from willutil import Bunch

def get_cli_args():
   opts = argparse.ArgumentParser(allow_abbrev=False)
   opts.add_argument('datafiles', nargs='*', type=str)
   opts.add_argument('--overwrite', action='store_true', default=False)
   opts = opts.parse_args(sys.argv[1:])
   return Bunch(opts)

def main():
   opts = get_cli_args()
   if not opts.datafiles:
      print('HARD-CODED DATA FILES')
      opts.datafiles.extend([
         # 'rpxdock/data/pdb_res_pair_data_si30_10.pickle',
         # './rpxdock/data/respairdat10.pickle',
         # './rpxdock/data/respairdat10_plus_xmap_rots.pickle',
         # 'rpxdock/data/pairscore10.pickle',
         # './rpxdock/data/hscore/small_ilv_h/old/pdb_res_pair_data_si30_10_rots_H_ILV_SSindep_p0.5_b1_hier3_Kflat_1_0.pickle',
         # './rpxdock/data/hscore/small_ilv_h/old/pdb_res_pair_data_si30_10_rots_H_ILV_SSindep_p0.5_b1_hier1_Kflat_1_0.pickle',
         # './rpxdock/data/hscore/small_ilv_h/old/pdb_res_pair_data_si30_10_rots_H_ILV_SSindep_p0.5_b1_hier4_Kflat_1_0.pickle',
         # './rpxdock/data/hscore/small_ilv_h/old/pdb_res_pair_data_si30_10_rots_H_ILV_SSindep_p0.5_b1_hier0_Kflat_1_0.pickle',
         # './rpxdock/data/hscore/small_ilv_h/old/pdb_res_pair_data_si30_10_rots_H_ILV_SSindep_p0.5_b1_base.pickle',
         # './rpxdock/data/hscore/small_ilv_h/old/pdb_res_pair_data_si30_10_rots_H_ILV_SSindep_p0.5_b1_hier2_Kflat_1_0.pickle',
         # 'rpxdock/data/testdata/test_plug_hier_trim.pickle',
         # 'rpxdock/data/testdata/test_cage_hier_3comp.pickle',
         # 'rpxdock/data/testdata/test_plug_hier.pickle',
         # 'rpxdock/data/testdata/test_make_cyclic_hier_trim.pickle',
         # 'rpxdock/data/testdata/test_make_cyclic_hier.pickle',
         # 'rpxdock/data/testdata/test_asym.pickle',
         # 'rpxdock/data/testdata/test_asym_trim.pickle',
         # 'rpxdock/data/testdata/test_plug_olig_grid.pickle',
         # 'rpxdock/data/testdata/test_make_cyclic_grid.pickle',
         # 'rpxdock/data/testdata/test_cage_hier_D3_onecomp_notrim.pickle',
         # 'rpxdock/data/testdata/test_cage_hier_onecomp_notrim.pickle',
         # 'rpxdock/data/testdata/test_cage_hier_no_trim.pickle',
         # 'rpxdock/data/testdata/test_cage_grid_onecomp_notrim.pickle',
         # 'rpxdock/data/testdata/test_cage_hier_D3_2_onecomp_notrim.pickle',
         # 'rpxdock/data/testdata/test_deepesh_1comp_bug.pickle',
         # 'rpxdock/data/testdata/test_layer_hier_3comp.pickle',
         # 'rpxdock/data/testdata/test_cage_hier_trim.pickle',
         # 'rpxdock/data/testdata/test_plug_olig_hier.pickle',
      ])
      opts.overwrite = True

   convert_rpx_data(**opts)
   print('DONE')

def convert_rpx_data(datafiles, **opts):
   for fname in datafiles:
      convert_rpxdata_datafile(fname, **opts)

def convert_rpxdata_datafile(fname, **opts):
   dat = rp.load(fname)

   if isinstance(dat, xr.Dataset) and 'r_pdbid' in dat:
      # new file name
      newfname = fname + '.nc'
      if fname.endswith('.pickle'):
         newfname = fname[:-7] + '.nc'
      # convert
      print(f'converting ResPairData {fname} to {newfname}')
      rp.motif.convert_respairdat_to_netcdf(dat, newfname, **opts)

   elif isinstance(dat, rp.motif.ResPairScore):
      newfname = fname + '.rpx.txz'
      if fname.endswith('.pickle'):
         newfname = fname[:-7] + '.rpx.txz'
      print(f'converting ResPairScore {fname} to {newfname}')
      rp.motif.respairscore_to_tarball(dat, newfname, **opts)

   elif isinstance(dat, rp.motif.Xmap):
      newfname = fname + '.xmap.txz'
      if fname.endswith('.pickle'):
         newfname = fname[:-7] + '.xmap.txz'
      print(f'converting Xmap {fname} to {newfname}')
      rp.motif.xmap_to_tarball(dat, newfname, **opts)

   elif isinstance(dat, rp.search.Result):
      newfname = fname + '.result.txz'
      if fname.endswith('.pickle'):
         newfname = fname[:-7] + '.result.txz'
      print(f'converting Xmap {fname} to {newfname}')
      rp.search.result_to_tarball(dat, newfname, *opts)
      # test = rp.search.result_from_tarball(newfname)
      # assert test.data == dat.data
   else:
      print(f'ignoring data type {type(dat)} from file {fname}')
      assert 0

if __name__ == '__main__':
   main()
