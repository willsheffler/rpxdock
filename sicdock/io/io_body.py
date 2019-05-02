import numpy as np

from sicdock.body import Body
from sicdock.io.io import *


def make_pdb_from_bodies(
    bodies,
    symframes=[np.eye],
    start=(0, 0),
    use_body_sym=False,
    keep=lambda x: True,
    no_duplicate_chains=False,
    no_duplicate_reschain_pairs=True,
    include_cen=True,
    only_atoms=None,
    chain_letters=all_pymol_chains,
):
    allatomnames = "N CA C O CB CEN".split()
    if not only_atoms and not include_cen:
        only_atoms = "N CA C O CB".split()
    if not only_atoms:
        only_atoms = allatomnames

    elems = "N C C O C X".split()
    aindex = [allatomnames.index(a) for a in only_atoms]
    if isinstance(chain_letters, int):
        chain_letters = all_pymol_chains[:chain_letters]

    startatm = start[0]
    startchain = start[1]
    bodies = [bodies] if isinstance(bodies, Body) else bodies
    s = ""
    ia = startatm
    max_resno = np.repeat(int(-9e9), len(chain_letters))
    for xsym in symframes:
        for body in bodies:
            com = xsym @ body.pos[:, 3]
            if not keep(com):
                continue
            crd = xsym @ body.positioned_coord(asym=not use_body_sym)[..., None]
            cen = xsym @ body.positioned_cen(asym=not use_body_sym)[..., None]
            nchain = len(np.unique(body.chain[: len(crd)]))
            for i in range(len(crd)):
                ic = body.chain[i] + startchain
                c = ic % len(chain_letters)
                aa = body.seq[i]
                resno = body.resno[i]
                if ic >= len(chain_letters):
                    if no_duplicate_chains:
                        break
                    if no_duplicate_reschain_pairs:
                        resno = max_resno[c] + 1
                max_resno[c] = max(resno, max_resno[c])
                cletter = chain_letters[c]
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
                    xyz = crd[i, j] if j != 5 else cen[i]
                    aname = allatomnames[j]
                    s += pdb_format_atom(
                        ia=ia + 1,
                        ir=resno + 1,
                        rn=aa,
                        xyz=xyz,
                        c=cletter,
                        an=aname,
                        occ=1,
                        elem=elems[j],
                    )
                    # print(s)
                    # return (s,)
                    ia += 1
            s += "TER\n"
            startchain += nchain
    startatm = ia
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


def dump_pdb_from_bodies(fname, bodies, symframes=[np.eye(4)], **kw):
    s, *_ = make_pdb_from_bodies(bodies, symframes, **kw)
    with open(fname, "w") as out:
        out.write(s)
