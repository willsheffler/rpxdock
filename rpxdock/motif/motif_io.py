import os, tempfile, json, tarfile, io, collections
import numpy as np
import xarray as xr
import rpxdock as rp

def save_bunch(bunch, path):
   nobunches = rp.util.bunch.unbunchify(bunch)
   with open(path, 'w') as out:
      json.dump(nobunches, out)

def load_bunch(bunch, inp):
   return rp.Bunch(json.load(inp))

def save_phmap(phmap, path):
   k, v = phmap.items_array()
   vtype = 'u' if v.dtype == np.dtype('u8') else 'f'
   with open(path + f'.PHMap_u8{vtype}8_keys.npy', 'wb') as out:
      np.save(out, k)
   with open(path + f'.PHMap_u8{vtype}8_vals.npy', 'wb') as out:
      np.save(out, v)

def save_xbin(xbin, path, lbl='xbin'):
   state = {
      lbl:
      dict(
         xbin_type=str(type(xbin)),
         xbin_cart_resl=xbin.cart_resl,
         xbin_ori_resl=xbin.ori_resl,
         xbin_cart_bound=xbin.max_cart,
      )
   }
   # print('XBIN SAVE')
   # for k, v in state[lbl].items():
   #    print('   ', k, type(v), v)
   with open(path, 'w') as out:
      json.dump(state, out)

def load_xbin(inp, lbl='xbin'):
   state = json.load(inp)[lbl]
   # print('XBIN LOAD')
   # for k, v in state.items():
   #    print('   ', k, type(v), v)

   xbin_type = state['xbin_type']
   if xbin_type.endswith("Xbin_float'>"):
      xbin = rp.xbin.Xbin_float(
         cart_resl=state['xbin_cart_resl'],
         ori_resl=state['xbin_ori_resl'],
         max_cart=state['xbin_cart_bound'],
      )
   elif xbin_type.endswith("Xbin_double'>"):
      xbin = rp.xbin.Xbin_double()
   else:
      raise TypeError(f'unknown Xbin variant {xbin_type}')
   # xbin.__setstate__((
   #    state['xbin_cart_resl'],
   #    state['xbin_ori_nside'],
   #    state['xbin_cart_bound'],
   # )) # this doesn't work and I dont know why

   return xbin

def save_xmap(xmap, path):
   save_xbin(xmap.xbin, path + '.xbin.json')
   save_phmap(xmap.phmap, path)
   save_bunch(xmap.attr, path + '.attr.json')

def respairscore_to_tarball(rps, fname, overwrite=False):
   if not fname.endswith(('.txz', '.tar.xz')):
      fname += '.txz'

   if os.path.exists(fname) and not overwrite:
      raise FileExistsError(f'file exists {fname}')

   with tempfile.TemporaryDirectory() as td:

      for mname in dir(rps):
         if mname.startswith('__'): continue
         member = getattr(rps, mname)

         if mname == 'rotchi':
            # replace rotchi with masked array
            rotchi = np.empty((len(member), 4), dtype='f8')
            rotchi.fill(np.nan)
            for i, chis in enumerate(member):
               for j, chi in enumerate(chis):
                  rotchi[i, j] = chi
            with open(td + '/rotchi.npy', 'wb') as out:
               np.save(out, rotchi, allow_pickle=False, fix_imports=False)

         elif mname == 'rotlbl':
            with open(td + '/rotlbl.json', 'w') as out:
               json.dump(member, out)

         elif mname == 'hier_maps':
            assert len(member) == 0, 'dont know how to serialize hier_maps, do individually'

         elif mname == 'rotspace':
            member = member.drop('dim_0')
            with open(td + '/' + mname + '.nc', 'wb') as out:
               member.to_netcdf(out)

         elif isinstance(member, np.ndarray):
            with open(td + '/' + mname + '.npy', 'wb') as out:
               np.save(out, member, allow_pickle=False, fix_imports=False)

         elif isinstance(member, dict):
            save_bunch(member, td + '/' + mname + '.json')

            # with open(td + '/' + mname + '.json', 'w') as out:
            # json.dump(nobunches, out)

         elif isinstance(member, rp.phmap.phmap.PHMap_u8u8):
            save_phmap(member, td + '/' + mname)

         elif isinstance(member, rp.motif.pairscore.Xmap):
            save_xmap(member, td + '/' + mname + '.Xmap')

         elif isinstance(member, rp.xbin.xbin.Xbin_float):
            save_xbin(member, td + '/' + mname + '.Xbin_float.json')

         elif str(type(member)) == "<class 'method'>":
            continue

         else:
            print('EROOR on member', mname, type(member))
            print(member)

      cmd = f'cd {td} && tar cjf {os.path.abspath(fname)} *'
      assert not os.system(cmd)
      return fname

def respairscore_from_tarball(fname):
   rps = rp.motif.ResPairScore()
   rps.hier_maps = []
   xmaps = collections.defaultdict(dict)
   phmaps = collections.defaultdict(dict)

   with tarfile.open(fname) as tar:
      # for m in tar.getmembers():
      # print(m.name)

      for m in tar.getmembers():
         raw = tar.extractfile(m)
         inp = io.BytesIO()
         inp.write(raw.read())
         inp.seek(0)

         if m.name.count('.Xmap.'):
            # print(m.name)
            mname, xm, f, ext = m.name.split('.')
            # print(f)
            assert xm == 'Xmap'
            if f == 'attr':
               assert ext == 'json'
               xmaps[mname][f] = json.load(inp)
            elif f.startswith('PHMap_u8f8_'):
               assert ext == 'npy'
               xmaps[mname][f] = np.load(inp)
            elif f == 'xbin':
               xmaps[mname][f] = load_xbin(inp)
            else:
               print(m.name)
               print(mname, f, ext)
               assert 0, 'Xmap madness in pairscore tarball!'

         elif m.name.endswith('.Xbin_float.json'):
            mname = m.name[:-16]
            setattr(rps, mname, load_xbin(inp))

         elif m.name.count('.PHMap_u8u8_'):
            mname, f, ext = m.name.split('.')
            assert ext == 'npy'
            phmaps[mname][f] = np.load(inp)

         elif m.name.endswith('.npy'):
            mname = m.name[:-4]
            ary = np.load(inp)
            setattr(rps, mname, ary)

         elif m.name.endswith('.json'):
            mname = m.name[:-5]
            val = json.load(inp)
            if isinstance(val, dict):
               val = val
            setattr(rps, mname, val)

         elif m.name.endswith('.nc'):
            mname = m.name[:-3]
            setattr(rps, mname, xr.open_dataset(inp))

         else:
            print('unk', m.name)
            assert 0

   # collect stored Xmaps
   for mname, xmdat in xmaps.items():
      assert len(xmdat) == 4
      k = xmdat['PHMap_u8f8_keys']
      v = xmdat['PHMap_u8f8_vals']
      phmap = rp.phmap.PHMap_u8f8()
      phmap[k] = v
      xmap = rp.motif.Xmap(xmdat['xbin'], phmap, xmdat['attr'])
      setattr(rps, mname, xmap)

   # collect stored PHMaps
   for mname, phdat in phmaps.items():
      assert len(phdat) == 2
      if 'PHMap_u8f8_keys' in phdat:
         phm = rp.phmap.PHMap_u8f8()
         k = phdat['PHMap_u8f8_keys']
         v = phdat['PHMap_u8f8_vals']
         assert k.dtype == np.dtype('u8')
         assert v.dtype == np.dtype('f8')
         phm[k] = v
         setattr(rps, mname, phm)
      elif 'PHMap_u8u8_keys' in phdat:
         phm = rp.phmap.PHMap_u8u8()
         k = phdat['PHMap_u8u8_keys']
         v = phdat['PHMap_u8u8_vals']
         assert k.dtype == np.dtype('u8')
         assert v.dtype == np.dtype('u8')
         phm[k] = v
         setattr(rps, mname, phm)
      else:
         assert 0, 'madness in stored respairscore phmap info'
   # add xmaps

   return rps
