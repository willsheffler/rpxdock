import copy, logging, os, tempfile, tarfile, io, json, itertools
from collections import OrderedDict, abc, defaultdict, Sequence
import numpy as np, rpxdock as rp
from rpxdock.util import sanitize_for_storage, num_digits
import rpxdock as rp
import willutil as wu

log = logging.getLogger(__name__)

class Result:
   def __init__(
      self,
      data_or_file=None,
      bodies=None,
      labels=None,
      pdb_extra_=None,
      **kw,
   ):
      import xarray as xr

      if data_or_file:
         assert len(kw) == 0
         if isinstance(data_or_file, xr.Dataset):
            self.data = data_or_file
         else:
            self.load(file_)
      else:
         attrs = OrderedDict(kw['attrs']) if 'attrs' in kw else None
         if attrs: del kw['attrs']
         attrs = sanitize_for_storage(attrs)
         self.data = xr.Dataset(dict(**kw), attrs=attrs)

      self.bodies, self.labels = process_body_labels(bodies, labels, self.data)
      self.pdb_extra_ = pdb_extra_

      self.xforms.shape[-2:] == (4, 4)
      assert len(self.data['scores']) == len(self.data['xforms'])

      # UGH.... GROSS.... b/c I always mistype these
      self.dump_pdb_top_score = self.dump_pdbs_top_score
      self.dump_pdb_top_score_each = self.dump_pdbs_top_score_each

   # @property
   # def bodies(self):
   #    return self._bodies

   # @property
   # def labels(self):
   #    return self._labels

   def sortby(self, *args, **kw):
      r = copy.copy(self)
      r.data = self.data.sortby(*args, **kw)
      return r

   def __getattr__(self, name):
      if name in ('data'):
         raise AttributeError
      return getattr(self.data, name)

   def __getitem__(self, name):
      return self.data[name]

   def __setitem__(self, name, val):
      self.data[name] = val

   def __str__(self):
      return "Result with data = " + str(self.data).replace("\n", "\n  ")

   def copy(self):
      return Result(self.data.copy())

   def __setstate__(self, state):
      # print(type(state))
      # print(list(state.keys()))
      self.data = state['data']
      bodies = state['bodies']
      try:
         labels = state['labels']
      except KeyError:
         labels = state['body_label_']
      self.pdb_extra_ = state['pdb_extra_']

      self.dump_pdb_top_score = self.dump_pdbs_top_score
      self.dump_pdb_top_score_each = self.dump_pdbs_top_score_each

      # self.bodies, self.labels = bodies, labels
      self.bodies, self.labels = process_body_labels(bodies, labels, self.data)

   # def getstate(self):
   #    return self.data.to_dict()

   # def setstate(self, state):
   #    import xarray as xr
   #    self.data = xr.Dataset.from_dict(state)

   def sel(self, *args, **kw):
      r = copy.copy(self)
      r.data = r.data.sel(*args, **kw)
      return r

   def dump_pdbs_top_score(self, nout_top=10, **kw):
      best = np.argsort(-self.scores)
      return self.dump_pdbs(best[:nout_top], lbl='top', **kw)

   def top_each(self, neach=1):
      which = dict()
      for ijob, imodel in self.scores.groupby(self.ijob).groups.items():
         imodel = np.array(imodel)
         ibest = np.argsort(-self.scores.data[imodel])
         which[ijob] = imodel[ibest[:neach]]
      return which

   def dump_pdbs_top_score_each(self, nout_each=1, **kw):
      if nout_each == 0: return
      which = self.top_each(nout_each)
      if not len(which):
         raise ValueError("can't dump pdbs, no results available")
      ndigijob = num_digits(len(which) - 1)
      dumped = set()
      ndigmdl = max(np.max(num_digits(v)) for v in which.values())
      for ijob, imodel in which.items():
         dumped |= self.dump_pdbs(imodel, lbl=f'job{ijob:0{ndigijob}}_top', ndigmdl=ndigmdl, **kw)
      return dumped

   def dump_pdbs(
      self,
      which='all',
      ndigwhich=None,
      ndigmdl=None,
      lbl='',
      skip=[],
      output_prefix='rpx',
      **kw,
   ):
      if isinstance(which, str):
         if which == 'all':
            which = range(len(self.data))
         else:
            raise ValueError(f'dump_pdbs only understandes "all" or sequence, not {which}')

      if len(which) == 0: return set()
      if isinstance(which, abc.Mapping):
         raise ValueError('dump_pdbs takes sequence not mapping')
      if 'fname' in kw and kw['fname'] is not None:
         raise ArgumentError('fname is not a valid argument for dump_pdbs, because multiple files')
      if not output_prefix and 'output_prefix' in self.attrs:
         output_prefix = self.output_prefix
      if ndigwhich is None: ndigwhich = num_digits(len(which) - 1)
      if ndigmdl is None: ndigmdl = num_digits(max(which))

      dumped = set()
      for i, imodel in enumerate(which):
         assert not isinstance(imodel, np.ndarray) or len(imodel) == 1
         if not imodel in skip:
            dumped.add(int(imodel))
            prefix_tmp = f'{output_prefix}_{lbl}{i:0{ndigwhich}}_{int(imodel):0{ndigmdl}}_'
            self.dump_pdb(imodel, output_prefix=prefix_tmp, **kw)
      return dumped

   def dump_pdb(
      self,
      imodel,
      output_prefix='',
      output_suffix='',
      fname=None,
      output_body='ALL',
      sym='',
      sep='_',
      skip=[],
      hscore=None,
      output_asym_only=False,
      output_closest_subunits=False,
      **kw,
   ):
      outfnames = list()
      if not sym and 'sym' in self.attrs: sym = self.attrs['sym']
      if not sym and 'sym' in self.data: sym = self.data.sym.data[imodel]
      sym = sym if sym else "C1"
      if not output_prefix and 'output_prefix' in self.attrs:
         output_prefix = self.output_prefix

      ijob = 0
      if 'ijob' in self.data:
         ijob = self.ijob[imodel].values

      bod = self.bodies[ijob]
      multipos = self.xforms.ndim == 4
      if multipos and self.xforms.shape[1] != len(bod):
         raise ValueError("number of positions doesn't match number of bodies")
      if str(output_body).upper() == 'ALL': output_body = list(range(len(bod)))
      if isinstance(output_body, int): output_body = [output_body]
      if not all(w < len(bod) for w in output_body):
         raise ValueError(f'output_body ouf of bounds {output_body}')
      bod = [bod[i] for i in output_body]
      if self.xforms.ndim == 4:
         for x, b in zip(self.xforms[imodel], bod):
            b.move_to(x.data)
      else:
         if isinstance(bod[0], list):
            bod[0][0].move_to(self.xforms[imodel].data)
         else:
            bod[0].move_to(self.xforms[imodel].data)
      if not fname:
         output_prefix = output_prefix + sep if output_prefix else ''
         body_names = [b.label for b in bod]
         if len(output_body) > 1 and self.labels:

            # print(output_body)
            # print(imodel.data)
            # print(len(self.labels))
            # print(self.labels)
            bodlab = [self.labels[ijob][z] for z in output_body]
            body_names = [bl + '_' + lbl for bl, lbl in zip(bodlab, body_names)]
         middle = '__'.join(body_names)
         output_suffix = sep + output_suffix if output_suffix else ''
         fname = output_prefix + middle + output_suffix + '.pdb'
      fname = os.path.abspath(fname)
      log.info(f'dumping pdb {fname} score {self.scores.data[imodel]}')
      bfactor = None
      # hscore scores residue pairs and puts bfactor in pdb
      if hscore and len(bod) == 2:
         sm = hscore.score_matrix_inter(
            bod[0],
            bod[1],
            symframes=rp.geom.symframes(sym, pos=self.xforms.data[imodel], **kw),
            wts=kw['wts'],
         )
         bfactor = [sm.sum(axis=1), sm.sum(axis=0)]
      bounds = np.tile([[-9e9], [9e9]], len(bod)).T
      if 'reslb' in self.data and 'resub' in self.data:
         bounds = np.stack([self.reslb[imodel], self.resub[imodel]], axis=-1)
      symframes = rp.geom.symframes(sym, pos=self.xforms.data[imodel], **kw)

      if output_asym_only and output_closest_subunits:
         if len(bod) == 2:
            best = 0, None, None
            x = self.xforms.data[imodel]
            bod[0].pos = x[0]
            for i, f in enumerate(symframes):
               bod[1].pos = f @ x[1]
               ctc = bod[0].contact_count(bod[1], maxdis=8)
               # print(i, ctc)
               if ctc > best[0]:
                  best = ctc, i, f
            # print('best', best)
            symframes = [np.eye(4)]
            bod[1].pos = best[2] @ self.xforms.data[imodel][1]
         elif len(bod) > 2:
            raise NotImplementedError
      elif output_asym_only:
         symframes = [np.eye(4)]

      outfnames.append(fname)
      print('dumping', fname)
      rp.io.dump_pdb_from_bodies(
         fname,
         bod,
         symframes=symframes,
         resbounds=bounds,
         bfactor=bfactor,
         **kw,
      )
      if hasattr(self, 'pdb_extra_') and self.pdb_extra_ is not None:
         with open(fname, 'a') as out:
            out.write(self.pdb_extra_[int(imodel)])
      if hasattr(self.data, 'helix_n_to_primary'):
         symframes = symframes[np.array(
            [0, self.data.helix_n_to_primary[imodel], self.data.helix_n_to_secondry[imodel]])]
         rp.io.dump_pdb_from_bodies(fname + '_hbase.pdb', bod, symframes=symframes,
                                    resbounds=bounds, bfactor=bfactor, **kw)
         # assert 0, 'testing helix dump'
      return outfnames

   def __len__(self):
      return len(self.model)

   @property
   def ndocks(self):
      return len(self.dockinfo)

   def __eq__(self, other):
      return self.data.equals(other.data)

def dict_coherent_entries(alldicts):
   sets = defaultdict(set)
   badkeys = set()
   for d in alldicts:
      for k, v in d.items():
         try:
            sets[k].add(v)
         except:
            badkeys.add(k)
   return {k: v.pop() for k, v in sets.items() if len(v) == 1 and not k in badkeys}

def concat_results(results, **kw):
   import xarray as xr
   if isinstance(results, Result): results = [results]
   assert len(results) > 0
   ijob = np.repeat(np.arange(len(results)), [len(r) for r in results])
   # assert all(r.labels == results[0].labels for r in results)
   allattrs = [r.attrs for r in results]
   common = dict_coherent_entries(allattrs)
   ret = Result(xr.concat([r.data for r in results], dim='model', **kw))
   # r.bodies = [r.bodies[0] for r in results]
   ret.bodies = list(itertools.chain(*(r.bodies for r in results)))
   ret.data['ijob'] = (['model'], ijob)
   ret.data.attrs = OrderedDict(dockinfo=allattrs, **common)
   ret.labels = list(itertools.chain(*(r.labels for r in results)))
   if results[0].pdb_extra_ is not None:
      ret.pdb_extra_ = list()
      for x in results:
         assert len(x.pdb_extra_) == len(x.data.scores)
         ret.pdb_extra_.extend(x.pdb_extra_)
   assert not isinstance(ret.bodies[0][0], list)
   assert not isinstance(ret.labels[0][0], list)
   return ret

def dummy_result_data(size=1000):
   size = size // 5 * 5
   from rpxdock.homog import rand_xform
   return wu.Bunch(
      ijob=(['model'], np.repeat([3, 1, 2, 4, 0], size / 5).astype('i8')),
      scores=(["model"], np.random.rand(size).astype('f4')),
      xforms=(["model", "hrow", "hcol"], rand_xform(size).astype('f4')),
      rpx_plug=(["model"], np.random.rand(size).astype('f4')),
      rpx_hole=(["model"], np.random.rand(size).astype('f4')),
      ncontact_plug=(["model"], np.random.rand(size).astype('f4')),
      ncontact_hole=(["model"], np.random.rand(size).astype('f4')),
      reslb=(["model"], np.random.randint(0, 100, size)),
      resub=(["model"], np.random.randint(100, 200, size)),
   )

def dummy_result(size=1000):
   return Result(**dummy_result_data(size))

def assert_results_close(r, s, n=-1, tol=1e-6, xtol=1e-3):
   if set(r.keys()) != set(s.keys()):
      print(list(r.keys()))
      print(list(s.keys()))
   # print(list(r.keys()))
   # print(list(s.keys()))
   assert set(r.keys()) == set(s.keys()), 'results must have same fields'
   assert np.allclose(r.scores[:n], s.scores[:n], atol=tol)
   assert np.allclose(r.xforms[:n], s.xforms[:n], atol=xtol)
   for k in r.data:
      if k in ('scores', 'xforms'): continue
      assert np.allclose(r[k][:n], s[k][:n], atol=tol)

def result_from_tarball(fname):
   import xarray as xr
   sources = 'no sources'
   bodies = list()
   labels = list()
   bodylabels = None
   pdb_extra_ = None
   data = None
   timer = wu.Timer()
   with tarfile.open(fname) as tar:

      timer.checkpoint('into tarball')

      for m in tar.getmembers():
         raw = tar.extractfile(m)
         inp = io.BytesIO()
         inp.write(raw.read())
         inp.seek(0)

         if m.name == 'dataset.nc':
            data = xr.open_dataset(inp)

         elif m.name == 'original_sources.txt':
            sources = inp.read().decode()

         elif m.name == 'body_labels.json':
            labels = json.loads(inp.read().decode())
            if isinstance(labels, dict):
               bodylabels = labels['bodylabels']
               labels = labels['resultlabels']

         elif m.name == 'pdb_extra_.json':
            pdb_extra_ = json.loads(inp.read().decode())

         elif m.name.endswith('.pdb') or m.name.endswith('.pdb.gz'):
            assert m.name.startswith('body_')
            i = int(m.name[5])
            j = int(m.name[7])
            # print('pdb', i, j, m.name)
            body = rp.Body(inp, source_filename=m.name)
            if len(bodies) <= i:
               bodies.append(list())
            bodies[i].append(body)
         else:
            assert 0, 'unknown result.txz member: ' + m.name
         timer.checkpoint(m.name)

   if sources == 'no sources':
      print('warning: result.txz file has no original_sources.txt')
   if len(bodies) == 0:
      print('warning: result.txz file has no body pdb files')
   if data is None:
      print('warning: result.txz file has no dataset.nc')

   if bodylabels:
      for ls, bs in zip(bodylabels, bodies):
         for l, b in zip(ls, bs):
            b.label = l

   result = rp.search.Result(
      data,
      bodies=bodies,
      labels=labels,
      pdb_extra_=pdb_extra_,
   )
   result.original_sources = sources
   timer.checkpoint('finish')
   # print(timer)
   return result

def result_to_tarball(result, fname, overwrite=False):

   if not fname.endswith(('.txz', '.tar.xz')):
      fname += '.txz'

   if os.path.exists(fname) and not overwrite:
      raise FileExistsError(f'file exists {fname}')

   if type(result) is not rp.search.Result:
      raise TypeError()

      # dictionaries / json no good for netcdf
   attrs = result.data.attrs
   attrs2 = dict()
   # attrs2 = sanitize_for_storage(attrs)
   # if 'arg' in attrs2:
   # del attrs2['arg']
   # if 'executor' in attrs2['arg']: del attrs2['arg']['executor']
   # if 'iface_summary' in attrs2['arg']: del attrs2['arg']['iface_summary']

   # print('------------------- orig --------------')
   # for k, v in attrs.items():
   #    print(k)
   #    print(str(v)[:100])
   #    print()

   # assert 0
   # result.data.attrs['dockinfo'] = repr(result.data.attrs['dockinfo'])
   # print(type(attrs2), type(attrs))
   if 'dockinfo' in attrs and len(attrs['dockinfo']):
      if 'arg' in attrs['dockinfo'][0]:
         attrs2['arg'] = repr(dict(attrs['dockinfo'][0]['arg']))
      for i, di in enumerate(attrs['dockinfo']):
         del di['arg']
      attrs2['dockinfo'] = repr(attrs['dockinfo'])

   for k, v in attrs.items():
      attrs2[k] = repr(v)

   # attrs2 = sanitize_for_storage(attrs2, netcdf=True)

   # print('---------------- new ----------------------')
   # for k, v in attrs2:
   #    print(k)
   #    print(str(v)[:100])
   #    print()

   result.data.attrs = attrs2

   # del result.data.attrs['dockinfo']
   # print(result.data.attrs['dockinfo'])
   # for di in result.data.attrs['dockinfo']:
   #    for k, v in di:
   #       print('   ', k)
   #       print('   ', v)
   #       print()
   #
   with tempfile.TemporaryDirectory() as td:

      with open(td + '/dataset.nc', 'wb') as out:
         result.data.to_netcdf(out)

      if isinstance(result.bodies[0][0], tuple):
         assert len(result.bodies) == 1
         result.bodies = result.bodies[0]

      sources = list()
      for i, bods in enumerate(result.bodies):
         for j, bod in enumerate(bods):
            sources.append(f'ijob {i} icomponent {j} source {bod.source()}')
            fn = f'body_{i}_{j}_{bod.source()}'.replace("/", "^")
            print('fname for tarball', fn)
            with open(td + '/' + fn, 'w') as out:
               pdbstr, _ = bod.str_pdb()
               out.write(pdbstr)

      with open(td + '/original_sources.txt', 'w') as out:
         out.write(os.linesep.join(sources))
      # print(os.listdir(td))

      if hasattr(result, 'pdb_extra_'):
         with open(td + '/pdb_extra_.json', 'w') as out:
            out.write(json.dumps(result.pdb_extra_))

      with open(td + '/body_labels.json', 'w') as out:
         body_labels = dict(bodylabels=[[b.label for b in bod] for bod in result.bodies],
                            resultlabels=result.labels)
         out.write(json.dumps(body_labels))

      cmd = f'cd {td} && tar cjf {os.path.abspath(fname)} *'
      assert not os.system(cmd)

   return fname

def process_body_labels(bodies, labels, data):
   njob = max(data.ijob.data,default=-1)+1 if 'ijob' in data else 1
   ndim = data['xforms'].shape[1] if data['xforms'].ndim == 4 else 1
   if bodies in ([], [[[]]]): bodies = None
   if labels in ([], [[[]]]): labels = None

   if not isinstance(bodies, (list, tuple)):
      bodies = [[bodies for i in range(ndim)] for j in range(njob)]
   elif not isinstance(bodies[0], (list, tuple)):
      bodies = [bodies for i in range(njob)]
   bodies = bodies.copy()

   if not isinstance(labels, (list, tuple)):
      labels = [[None for i in bodies[j]] for j in range(njob)]
   elif not isinstance(labels[0], (list, tuple)):
      labels = [labels for i in range(njob)]
   # elif len(labels) == ndim:
   # labels = [njob * labels]
   labels = labels.copy()

   for i, l in enumerate(labels):
      for j, m in enumerate(l):
         m = m or 'body_job%i_comp%i' % (i, j)
         labels[i][j] = str(m)

   assert isinstance(bodies, (list, tuple))
   assert isinstance(bodies[0], (list, tuple))
   assert np.array(bodies).ndim == 2
   assert np.array(bodies).shape == np.array(labels).shape

   #print(bodies)
   #print(len(bodies))
   #print(data.ijob.data)
   #print(njob)

   if bodies and njob > 0:
      assert len(bodies) == njob
      assert len(bodies[0]) == ndim or (ndim == 1 and len(bodies[0]) == 2)

      if bodies[0]:
         # print(bodies)
         # print(type(bodies[0]))
         # print(type(bodies[0][0]))
         if not isinstance(bodies[0][0], (rp.Body, type(None))):
            raise TypeError(f'bodies must be type rp.Body or None, not {type(bodies[0][0])}')

   return bodies, labels

# github.com/minkbaek/BFF
# https://github.com/minkbaek/BFF/blob/main/rf_diffusion/TEST_COMMANDS