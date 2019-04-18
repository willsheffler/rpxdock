import numpy as np

from sicdock.body import Body


def pdb_format_atom(
    ia=0,
    an="ATOM",
    idx=" ",
    rn="RES",
    c="A",
    ir=0,
    insert=" ",
    x=0,
    y=0,
    z=0,
    occ=0,
    b=0,
    xyz=None,
):
    if xyz is not None:
        x, y, z, *_ = xyz.squeeze()
    if rn in aa1:
        rn = aa123[rn]
    if not isinstance(c, str):
        c = _chains[c]
    return _pdb_atom_record_format.format(**locals())


def make_pdb(
    bodies,
    symframes=[np.eye],
    start=None,
    use_body_sym=False,
    keep=lambda x: True,
    no_duplicate_chains=False,
):
    start = [0, 0] if start is None else start
    bodies = [bodies] if isinstance(bodies, Body) else bodies
    names = "N CA C O CB CEN".split()
    s = ""
    ia = start[0]
    for xsym in symframes:
        for body in bodies:
            com = xsym @ body.pos[:, 3]
            if not keep(com):
                continue
            crd = xsym @ body.positioned_coord(asym=not use_body_sym)[..., None]
            cen = xsym @ body.positioned_cen(asym=not use_body_sym)[..., None]
            nchain = len(np.unique(body.chain[: len(crd)]))
            for i in range(len(crd)):
                ic = body.chain[i] + start[1]
                if no_duplicate_chains and ic >= len(_chains):
                    break
                c = ic % len(_chains)
                j = body.resno[i]
                aa = body.seq[i]
                s += pdb_format_atom(ia=ia + 0, ir=j, rn=aa, xyz=crd[i, 0], c=c, an="N")
                s += pdb_format_atom(
                    ia=ia + 1, ir=j, rn=aa, xyz=crd[i, 1], c=c, an="CA"
                )
                s += pdb_format_atom(ia=ia + 2, ir=j, rn=aa, xyz=crd[i, 2], c=c, an="C")
                s += pdb_format_atom(ia=ia + 3, ir=j, rn=aa, xyz=crd[i, 3], c=c, an="O")
                s += pdb_format_atom(
                    ia=ia + 4, ir=j, rn=aa, xyz=crd[i, 4], c=c, an="CB"
                )
                s += pdb_format_atom(ia=ia + 5, ir=j, rn=aa, xyz=cen[i], c=c, an="CEN")
                ia += 6
            start[1] += nchain
    start[0] = ia
    if start[1] > len(_chains) and not no_duplicate_chains:
        print(
            "WARNING: too many chains for a pdb",
            start[1] - len(_chains),
            "will be duplicates",
        )
    return s, start


def dump_pdb(fname, bodies, symframes=[np.eye(4)], **kw):
    s, *_ = make_pdb(bodies, symframes, **kw)
    with open(fname, "w") as out:
        out.write(s)


_pdb_atom_record_format = (
    "ATOM  {ia:5d} {an:^4}{idx:^1}{rn:3s} {c:1}{ir:4d}{insert:1s}   "
    "{x:8.3f}{y:8.3f}{z:8.3f}{occ:6.2f}{b:6.2f}\n"
)

aa1 = "ACDEFGHIKLMNPQRSTVWY"
aa123 = dict(
    A="ALA",
    C="CYS",
    D="ASP",
    E="GLU",
    F="PHE",
    G="GLY",
    H="HIS",
    I="ILE",
    K="LYS",
    L="LEU",
    M="MET",
    N="ASN",
    P="PRO",
    Q="GLN",
    R="ARG",
    S="SER",
    T="THR",
    V="VAL",
    W="TRP",
    Y="TYR",
)
aa321 = dict(
    ALA="A",
    CYS="C",
    ASP="D",
    GLU="E",
    PHE="F",
    GLY="G",
    HIS="H",
    ILE="I",
    LYS="K",
    LEU="L",
    MET="M",
    ASN="N",
    PRO="P",
    GLN="Q",
    ARG="R",
    SER="S",
    THR="T",
    VAL="V",
    TRP="W",
    TYR="Y",
)

_chains = (
    "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789abcdefghijklmnopqrstuvwxyz!@#$&.<>?]|-_\\~=%"
)
