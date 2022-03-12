import argparse, sys
import numpy as np
import xarray as xr
import rpxdock as rp

def get_cli_args():
   opts = argparse.ArgumentParser(allow_abbrev=False)
   opts.add_argument('datafiles', nargs='*', type=str)
   opts = opts.parse_args(sys.argv[1:])
   return rp.Bunch(opts)

def main():
   opts = get_cli_args()
   print(opts.datafiles)
   if not opts.datafiles:
      print('HARD-CODED DATA FILES')
      opts.datafiles.extend([
         './rpxdock/data/respairdat10.pickle',
         './rpxdock/data/respairdat10_plus_xmap_rots.pickle',
      ])
   for fname in opts.datafiles:
      convert_rpxdata_to_netcdf(fname)
   print('DONE')

def convert_rpxdata_to_netcdf(fname):
   dat = rp.load(fname)
   if isinstance(dat, xr.Dataset) and 'r_pdbid' in dat:
      # new file name
      newfname = fname + '.nc'
      if fname.endswith('.pickle'):
         newfname = fname.rstrip('.pickle') + '.nc'
      # convert
      print('converting', newfname)
      convert_respairdat_to_netcdf(dat, newfname)
   else:
      print('ignoring data from file', fname)

def convert_respairdat_to_netcdf(rpd, newfname):
   assert newfname.endswith('.nc')

   # print(rpd)

   # replace weights with key/val lists
   rpd.attrs['eweights_k'] = list(rpd.attrs['eweights'].keys())
   rpd.attrs['eweights_v'] = list(rpd.attrs['eweights'].values())
   del rpd.attrs['eweights']

   # replace rotchi with masked array
   rotchi = np.empty((len(rpd.attrs['rotchi']), 4), dtype='f8')
   rotchi.fill(np.nan)
   for i, chis in enumerate(rpd.attrs['rotchi']):
      for j, chi in enumerate(chis):
         rotchi[i, j] = chi
   rpd.attrs['rotchi'] = rotchi.reshape(-1)

   # replace chain lists with masked array
   maxchains = max(len(_.data.item()) for _ in rpd.chains)
   newchains = np.empty((len(rpd.chains), maxchains, 2), dtype='i')
   newchains.fill(np.nan)
   for i, ch in enumerate(rpd.chains):
      for j, be in enumerate(ch.item()):
         newchains[i, j] = be
   rpd = rpd.drop('chains')
   rpd = rpd.assign(chains=xr.Variable(('pdbid', 'nchain', 'chainbegend'), newchains))
   # print(rpd.chains.shape, rpd.chains.dtype, rpd.chains.dims)

   # assert 0
   stub = rpd.stub.data
   # print(rpd.stub.shape, rpd.stub.dtype, rpd.stub.dims)
   rpd = rpd.drop('stub')
   rpd = rpd.assign(stub=xr.Variable(('resid', 'hrow', 'hcol'), stub))
   # print(rpd.stub.shape, rpd.stub.dtype, rpd.stub.dims)

   rpd.to_netcdf(newfname)

   # print('DONE!')

if __name__ == '__main__':
   main()
