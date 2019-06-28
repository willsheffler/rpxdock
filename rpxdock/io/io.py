import numpy as np

all_pymol_chains = ("ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789abcdefghijklmnopqrstuvwxyz")

def pdb_format_atom(ia=0, an="ATOM", idx=" ", rn="RES", c="A", ir=0, insert=" ", x=0, y=0, z=0,
                    occ=1, b=1, elem=" ", xyz=None):
   if xyz is not None:
      x, y, z, *_ = xyz.squeeze()
   if rn in aa1:
      rn = aa123[rn]
   if not isinstance(c, str):
      c = all_pymol_chains[c]

   format_str = _pdb_atom_record_format
   if ia >= 100000:
      format_str = format_str.replace("ATOM  {ia:5d}", "ATOM {ia:6d}")
   if ir >= 10000:
      format_str = format_str.replace("{ir:4d}{insert:1s}", "{ir:5d}")

   return format_str.format(**locals())

def dump_pdb_from_points(fname, pts):
   with open(fname, "w") as out:
      for i, p in enumerate(pts):
         s = pdb_format_atom(x=p[0], y=p[1], z=p[2], ir=i)
         out.write(s)

_pdb_atom_record_format = ("ATOM  {ia:5d} {an:^4}{idx:^1}{rn:3s} {c:1}{ir:4d}{insert:1s}   "
                           "{x:8.3f}{y:8.3f}{z:8.3f}{occ:6.2f}{b:6.2f}           {elem:1s}\n")

aa1 = "ACDEFGHIKLMNPQRSTVWY"
aa123 = dict(A="ALA", C="CYS", D="ASP", E="GLU", F="PHE", G="GLY", H="HIS", I="ILE", K="LYS",
             L="LEU", M="MET", N="ASN", P="PRO", Q="GLN", R="ARG", S="SER", T="THR", V="VAL",
             W="TRP", Y="TYR")
aa321 = dict(ALA="A", CYS="C", ASP="D", GLU="E", PHE="F", GLY="G", HIS="H", ILE="I", LYS="K",
             LEU="L", MET="M", ASN="N", PRO="P", GLN="Q", ARG="R", SER="S", THR="T", VAL="V",
             TRP="W", TYR="Y")

def aname_to_elem(aname):
   "return based on first occurance of element letter"
   aname = aname.upper()
   elems = "COHNS"
   pos = [aname.find(e) for e in elems]
   poselem = sorted([(p, e) for p, e in zip(pos, elems) if p >= 0])
   return poselem[0][1]
