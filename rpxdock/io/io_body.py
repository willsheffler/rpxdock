import numpy as np

from rpxdock.body import Body
from rpxdock.io.io import *

def make_pdb_from_bodies(
   bodies,
   symframes=None,
   start=(0, 0),
   use_body_sym=None,
   keep=lambda x: True,
   no_duplicate_chains=False,
   no_duplicate_reschain_pairs=True,
   include_cen=True,
   only_atoms=None,
   chain_letters=None,
   resbounds=[],
   bfactor=None,
   occupancy=None,
   use_orig_coords=False,
   warn_on_chain_overflow=True,
   to_string=False,
   **kw,
):
   if symframes is None and use_body_sym is None:
      use_body_sym = True
      symframes = [np.eye(4)]
   elif symframes is None:
      symframes = [np.eye(4)]
   elif use_body_sym is None:
      use_body_sym = False

   allatomnames = "N CA C O CB CEN".split()
   if not only_atoms and not include_cen:
      only_atoms = "N CA C O CB".split()
   if not only_atoms:
      only_atoms = allatomnames

   elems = "N C C O C X".split()
   aindex0 = [allatomnames.index(a) for a in only_atoms]
   if chain_letters is None:
      n = len(all_pymol_chains) - len(all_pymol_chains) % len(bodies)
      chain_letters = all_pymol_chains[:n]
   elif isinstance(chain_letters, int):
      chain_letters = all_pymol_chains[:chain_letters]

   if bfactor and len(bfactor) > 9: bfactor = [bfactor]  # hacky

   if len(resbounds) == 0:
      resbounds = [(-9e9, 9e9)] * len(bodies)
   if isinstance(resbounds, dict):
      resbounds = np.stack([resbounds['reslb'], resbounds['resub']], axis=-1)
   if isinstance(resbounds[0], (int, np.int32, np.int64)):
      resbounds = [resbounds]

   startatm = start[0]
   startchain = start[1]
   bodies = [bodies] if isinstance(bodies, Body) else bodies
   s = ""
   ia = startatm
   # max_resno = np.repeat(int(-9e9), len(chain_letters))
   for isym, xsym in enumerate(symframes):
      for ibody, body in enumerate(bodies):
         com = xsym @ body.pos[:, 3]
         if not keep(com):
            continue
         crd = xsym @ body.positioned_coord(asym=not use_body_sym)[..., None]
         cen = xsym @ body.positioned_cen(asym=not use_body_sym)[..., None]
         orig_coords = body.positioned_orig_coords()
         nchain = len(np.unique(body.chain[:len(crd)]))
         reslb, resub = -9e9, 9e9
         if len(resbounds) > ibody:
            reslb, resub = resbounds[ibody][0], resbounds[ibody][1]
         for i in range(len(crd)):
            iasym = i % body.asym_body.nres if use_body_sym else i
            if not reslb <= i <= resub:
               continue
            ic = body.chain[i] + startchain
            c = ic % len(chain_letters)
            aa = body.seq[i]
            resno = body.resno[i]
            if ic >= len(chain_letters):
               if no_duplicate_chains:
                  break
               ic -= len(chain_letters)

            #    if no_duplicate_reschain_pairs:
            #       resno = max_resno[c] + 1
            # max_resno[c] = max(resno, max_resno[c])
            cletter = chain_letters[c]
            occ = 1 if occupancy is None else occupancy[resno]
            bfac = 0 if bfactor is None else bfactor[ibody][resno % len(bfactor[ibody])]
            # if bfac != 0 and isym == 0: print(isym, ibody, resno, bfac)
            aindex = range(len(orig_coords[i])) if use_orig_coords else aindex0
            for j in aindex:

               # ATOM      0  N   MET A   0      20.402  18.063   8.049  1.00  0.00           C
               # ATOM      1  N   SER A   1     -16.269 -14.208 -11.256  1.00 37.12           N
               # ATOM      2  CA  SER A   1     -17.067 -13.045 -10.756  1.00 35.28           C
               # ATOM      3  C   SER A   1     -16.360 -12.285  -9.626  1.00 31.65           C
               # ATOM      4  O   SER A   1     -16.027 -11.134  -9.821  1.00 31.56           O
               # ATOM      5  CB  SER A   1     -18.501 -13.427 -10.360  1.00 37.46           C
               # vs
               # ATOM      0  N   MET A   0      20.402  18.063   8.049  0.00  0.00
               # ATOM      1  CA  MET A   0      20.402  18.063   8.049  0.00  0.00
               # ATOM      2  C   MET A   0      20.402  18.063   8.049  0.00  0.00
               # ATOM      3  O   MET A   0      20.402  18.063   8.049  0.00  0.00
               # ATOM      4  CB  MET A   0      20.402  18.063   8.049  0.00  0.00
               if use_orig_coords:
                  xyz = xsym @ orig_coords[i][j]
                  aname = body.orig_anames[i][j]
                  elem = aname_to_elem(aname)
               else:
                  xyz = crd[i, j] if j != 5 else cen[i]
                  aname = allatomnames[j]
                  elem = elems[j]
               s += pdb_format_atom(
                  ia=ia + 1,
                  ir=resno + 1,
                  rn=aa,
                  xyz=xyz,
                  c=cletter,
                  an=aname,
                  occ=occ,
                  b=bfac,
                  elem=elem,
               )
               # print(s)
               # return (s,)
               ia += 1
         s += "TER\n"
         startchain += nchain
   startatm = ia
   if warn_on_chain_overflow:
      if start[1] > len(chain_letters):
         if no_duplicate_chains:
            print(
               "WARNING: too many chains for a pdb",
               start[1] - len(chain_letters),
               "of",
               len(chain_letters),
               "duplicates removed",
            )
         elif no_duplicate_reschain_pairs:
            print(
               "WARNING: too many chains for a pdb",
               start[1] - len(chain_letters),
               "of",
               len(chain_letters),
               "duplicate chains with offset resi",
            )
         else:
            print(
               "WARNING: too many chains for a pdb",
               start[1] - len(chain_letters),
               "of",
               len(chain_letters),
               "will be duplicate chain/resi pairs",
            )
   return s, (startatm, startchain)

def dump_pdb_from_bodies(fname, *args, **kw):
   s, *_ = make_pdb_from_bodies(*args, **kw)
   if not kw["to_string"]:
      with open(fname, "w") as out:
         out.write(s)
   else:
      return s
