_pdb_atom_record_format = (
    "ATOM  {atomi:5d} {atomn:^4}{idx:^1}{resn:3s} {chain:1}{resi:4d}{insert:1s}   "
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


def pdb_format_atom(
    atomi=0,
    atomn="ATOM",
    idx=" ",
    resn="RES",
    chain="A",
    resi=0,
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
    if resn in aa1:
        resn = aa123[resn]
    return _pdb_atom_record_format.format(**locals())
