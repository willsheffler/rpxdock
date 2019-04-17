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

_chains = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
_chains += _chains.lower() + "0123456789"


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
